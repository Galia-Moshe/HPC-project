/*
 * Exercise 3 - Ring All-Reduce over RDMA (Verbs API)
 * ---------------------------------------------------
 * A single-threaded C application implementing a process-group collective
 * library (Reduce-Scatter / All-Gather / All-Reduce) over a ring topology
 * of RC Queue Pairs, using RDMA Write with Immediate for signalling and
 * zero-copy RDMA Write for the All-Gather phase.
 *
 * Public API (as required by the exercise):
 *   int connect_process_group(char *servername, void **pg_handle);
 *   int pg_all_reduce(void *sendbuf, void *recvbuf, int count,
 *                      DATATYPE datatype, OPERATION op, void *pg_handle);
 *   int pg_close(void *pg_handle);
 *
 * CLI:
 *   test -myindex <1-based index> -list <host1> <host2> ... [-n iters]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <infiniband/verbs.h>

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define MAX_PROCESSES   4
#define TCP_BASE_PORT   18800   /* rank r listens on TCP_BASE_PORT + r  */
#define IB_PORT_DEFAULT 1
#define CQ_SIZE         512
#define SQ_DEPTH        64
#define RQ_DEPTH        64
#define WC_BATCH        16
/* TEMP (-tune-pipe): PIPE_SEGS is a runtime variable while tuning.
 * When done, delete these two lines and restore:
 *   #define PIPE_SEGS       4       (pipeline segments per ring chunk) */
static int g_pipe_segs = 16;
#define PIPE_SEGS       g_pipe_segs
#define CONNECT_RETRIES 200     /* ~20s of retries at 100ms backoff     */

/* Eager protocol: persistent stage buffer, sized generously enough to hold
 * one reduce-scatter chunk for anything the auto-tuner will ever try. Since
 * N>=2, a chunk is at most half the total message, so sizing this to the
 * auto-tuner's max message size covers every N. */
#define EAGER_STAGE_BYTES (4 * 1024 * 1024)

/* Eager/Rendezvous auto-tuning sweep range and per-size sample counts. */
#define AUTOTUNE_MIN_BYTES 4
#define AUTOTUNE_MAX_BYTES (4 * 1024 * 1024)
#define AUTOTUNE_WARMUP    5
#define AUTOTUNE_TRIALS    100

#define WARMUP_ITERS 5   /* main()'s pre-benchmark warmup, ask #1 */

#define BENCH_ITERS  1000   /* timed pg_all_reduce() calls per size in main() */
/* ------------------------------------------------------------------ */
/* Public collective-op types                                         */
/* ------------------------------------------------------------------ */

typedef enum { DT_INT32 = 0, DT_FLOAT = 1, DT_DOUBLE = 2 } DATATYPE;
typedef enum { OP_SUM = 0, OP_MAX = 1, OP_MIN = 2, OP_PROD = 3 } OPERATION;

/* ------------------------------------------------------------------ */
/* Wire-format structs exchanged over the TCP bootstrap channel        */
/* ------------------------------------------------------------------ */

/* QP connection parameters exchanged once, at ring-connect time */
struct qp_info {
    uint32_t     qpn;
    uint32_t     psn;
    uint16_t     lid;
    union ibv_gid gid;
};

/* Remote-memory descriptor exchanged once per pg_all_reduce phase,
 * so that a rank knows where in its neighbour's memory it may RDMA-Write. */
struct buf_info {
    uint64_t addr;
    uint32_t rkey;
    uint32_t size;
};

/* ------------------------------------------------------------------ */
/* pg_handle: the custom struct required by the exercise               */
/* ------------------------------------------------------------------ */

typedef struct {
    int   rank;
    int   world_size;
    int   next_rank;
    int   prev_rank;
    char  **hostnames;

    struct ibv_context *ibctx;
    struct ibv_pd       *pd;
    struct ibv_cq        *cq;

    struct ibv_qp *qp_next;   /* I RDMA-Write into next-rank's memory  */
    struct ibv_qp *qp_prev;   /* prev-rank RDMA-Writes into my memory  */

    int          ib_port;
    enum ibv_mtu mtu;
    int          sgid_idx;    /* -1 => pure IB / LID based, no GID     */

    int sock_next;            /* TCP control channel, client to next   */
    int sock_prev;            /* TCP control channel, server from prev */

    /* --- Eager protocol state (two-sided Send/Recv, no TCP handshake) --- */
    void          *eager_stage;       /* persistent reduce-scatter landing buf */
    struct ibv_mr *eager_stage_mr;
    size_t         eager_stage_bytes;

    /* Small cache so repeated calls with the SAME sendbuf/recvbuf (the
     * common case: a training loop reusing its gradient buffers) don't pay
     * ibv_reg_mr() on every single call. */
    void          *eager_mr_ptr;
    size_t         eager_mr_size;
    struct ibv_mr *eager_mr;

    /* Router threshold: pg_all_reduce() sends messages smaller than this
     * (in bytes) via the Eager path, and this-or-larger via Rendezvous.
     * Set by pg_autotune_threshold(); defaults to 0 (always Rendezvous)
     * until that has run. */
    size_t eager_threshold_bytes;
} pg_context_t;

/* ------------------------------------------------------------------ */
/* Global topology config, populated from -myindex / -list before      */
/* connect_process_group() is called (see main() below). The exercise  */
/* fixes connect_process_group()'s signature to a single `servername`  */
/* string, so the ring roster is threaded through here instead.        */
/* ------------------------------------------------------------------ */

static struct {
    int  my_index;      /* 1-based, as given on the command line */
    int  world_size;
    char *hostnames[MAX_PROCESSES];
} g_config = { 0, 0, {0} };

static void pg_set_topology(int my_index, char **hostlist, int world_size)
{
    g_config.my_index   = my_index;
    g_config.world_size = world_size;
    for (int i = 0; i < world_size; i++)
        g_config.hostnames[i] = strdup(hostlist[i]);
}

/* ------------------------------------------------------------------ */
/* Datatype / reduction helpers                                        */
/* ------------------------------------------------------------------ */

static size_t dt_size(DATATYPE dt)
{
    switch (dt) {
    case DT_INT32: return sizeof(int32_t);
    case DT_FLOAT: return sizeof(float);
    case DT_DOUBLE: return sizeof(double);
    default: return 0;
    }
}

#define OP_SUM_F(a, b)  ((a) + (b))
#define OP_MAX_F(a, b)  ((a) > (b) ? (a) : (b))
#define OP_MIN_F(a, b)  ((a) < (b) ? (a) : (b))
#define OP_PROD_F(a, b) ((a) * (b))

#define REDUCE_LOOP(T, EXPR) do {                              \
        T *d = (T *) dst;                                      \
        const T *s = (const T *) src;                          \
        for (int _i = 0; _i < n; _i++) d[_i] = EXPR(d[_i], s[_i]); \
    } while (0)

/* dst[i] = dst[i] OP src[i], for n elements of type dt. Applied by the
 * CPU as each pipelined segment lands during the Reduce-Scatter phase. */
static void reduce_apply(void *dst, const void *src, int n, DATATYPE dt, OPERATION op)
{
    switch (dt) {
    case DT_INT32:
        switch (op) {
        case OP_SUM:  REDUCE_LOOP(int32_t, OP_SUM_F);  break;
        case OP_MAX:  REDUCE_LOOP(int32_t, OP_MAX_F);  break;
        case OP_MIN:  REDUCE_LOOP(int32_t, OP_MIN_F);  break;
        case OP_PROD: REDUCE_LOOP(int32_t, OP_PROD_F); break;
        }
        break;
    case DT_FLOAT:
        switch (op) {
        case OP_SUM:  REDUCE_LOOP(float, OP_SUM_F);  break;
        case OP_MAX:  REDUCE_LOOP(float, OP_MAX_F);  break;
        case OP_MIN:  REDUCE_LOOP(float, OP_MIN_F);  break;
        case OP_PROD: REDUCE_LOOP(float, OP_PROD_F); break;
        }
        break;
    case DT_DOUBLE:
        switch (op) {
        case OP_SUM:  REDUCE_LOOP(double, OP_SUM_F);  break;
        case OP_MAX:  REDUCE_LOOP(double, OP_MAX_F);  break;
        case OP_MIN:  REDUCE_LOOP(double, OP_MIN_F);  break;
        case OP_PROD: REDUCE_LOOP(double, OP_PROD_F); break;
        }
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Ring chunk / pipeline-segment partitioning (pure functions of their  */
/* inputs, so sender and receiver independently compute identical      */
/* byte ranges without any extra metadata exchange).                   */
/* ------------------------------------------------------------------ */

static void chunk_range(int count, int N, int idx, int *start, int *len)
{
    int base = count / N, rem = count % N;
    if (idx < rem) { *len = base + 1; *start = idx * (base + 1); }
    else            { *len = base;     *start = rem * (base + 1) + (idx - rem) * base; }
}

static void seg_range(int chunk_len, int nsegs, int seg_i, int *start, int *len)
{
    int base = chunk_len / nsegs, rem = chunk_len % nsegs;
    if (seg_i < rem) { *len = base + 1; *start = seg_i * (base + 1); }
    else              { *len = base;     *start = rem * (base + 1) + (seg_i - rem) * base; }
}

/* ------------------------------------------------------------------ */
/* TCP bootstrap helpers (used only to exchange QP/MR metadata; all    */
/* payload movement happens over RDMA).                                */
/* ------------------------------------------------------------------ */

static int send_all(int fd, const void *buf, size_t len)
{
    const char *p = buf; size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, p + off, len - off);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) return -1;
        off += (size_t) n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len)
{
    char *p = buf; size_t off = 0;
    while (off < len) {
        ssize_t n = read(fd, p + off, len - off);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) return -1;
        off += (size_t) n;
    }
    return 0;
}

static int tcp_listen(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t) port);

    if (bind(fd, (struct sockaddr *) &addr, sizeof addr) < 0) { close(fd); return -1; }
    if (listen(fd, 4) < 0) { close(fd); return -1; }
    return fd;
}

static int tcp_connect_retry(const char *host, int port, int max_attempts)
{
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof hints);
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char portstr[16];
    snprintf(portstr, sizeof portstr, "%d", port);

    for (int attempt = 0; attempt < max_attempts; attempt++) {
        if (getaddrinfo(host, portstr, &hints, &res) == 0) {
            for (struct addrinfo *t = res; t; t = t->ai_next) {
                int fd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, t->ai_addr, t->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                close(fd);
            }
            freeaddrinfo(res);
        }
        usleep(100000); /* 100ms backoff: peer may not be listening yet */
    }
    return -1;
}

/* One control channel per ring edge is kept open for the process
 * group's lifetime; used here to advertise "where you may RDMA-Write
 * into me" ahead of every collective phase. */
static int exchange_landing_info(pg_context_t *ctx,
                                  const struct buf_info *my_landing,
                                  struct buf_info *next_landing_out)
{
    if (send_all(ctx->sock_prev, my_landing, sizeof *my_landing)) return -1;
    if (recv_all(ctx->sock_next, next_landing_out, sizeof *next_landing_out)) return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* QP setup helpers                                                    */
/* ------------------------------------------------------------------ */

static struct ibv_qp *create_ring_qp(pg_context_t *ctx)
{
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.send_cq = ctx->cq;
    attr.recv_cq = ctx->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr  = SQ_DEPTH;
    attr.cap.max_recv_wr  = RQ_DEPTH;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    return ibv_create_qp(ctx->pd, &attr);
}

/* Both ring QPs (qp_next and qp_prev) are created identically and both go
 * through this same INIT step, so both end up with the same access flags --
 * REMOTE_READ/REMOTE_WRITE so the QP will accept the peer's incoming RDMA
 * Read/Write, plus LOCAL_WRITE. Note qp_access_flags is only ever set here,
 * at INIT: the RTR/RTS transitions below (modify_qp_to_rtr/rts) carry no
 * IBV_QP_ACCESS_FLAGS bit and don't touch it -- access permissions for an RC
 * QP are fixed once at INIT and simply carry forward through the rest of
 * the state machine. */
static int modify_qp_to_init(struct ibv_qp *qp, int port)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = (uint8_t) port;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_LOCAL_WRITE;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int modify_qp_to_rtr(struct ibv_qp *qp, int port, enum ibv_mtu mtu,
                             int sgid_idx, const struct qp_info *dest)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = mtu;
    attr.dest_qp_num        = dest->qpn;
    attr.rq_psn             = dest->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer      = 12;
    attr.ah_attr.is_global     = 0;
    attr.ah_attr.dlid          = dest->lid;
    attr.ah_attr.sl            = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num      = (uint8_t) port;

    if (dest->gid.global.interface_id) {
        attr.ah_attr.is_global      = 1;
        attr.ah_attr.grh.hop_limit  = 1;
        attr.ah_attr.grh.dgid       = dest->gid;
        attr.ah_attr.grh.sgid_index = (uint8_t) sgid_idx;
    }

    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int modify_qp_to_rts(struct ibv_qp *qp, uint32_t my_psn)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout        = 14;
    attr.retry_cnt       = 7;
    attr.rnr_retry        = 7;
    attr.sq_psn            = my_psn;
    attr.max_rd_atomic      = 1;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

/* ------------------------------------------------------------------ */
/* WR posting / completion draining                                    */
/* ------------------------------------------------------------------ */

static int post_write_imm(struct ibv_qp *qp, uint64_t laddr, uint32_t lkey, uint32_t len,
                           uint64_t raddr, uint32_t rkey, uint32_t imm, uint64_t wr_id)
{
    struct ibv_sge sge = { .addr = laddr, .length = len, .lkey = lkey };
    struct ibv_send_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id      = wr_id;
    wr.opcode     = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data   = htonl(imm);
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey        = rkey;
    return ibv_post_send(qp, &wr, &bad);
}

/* RDMA-Write-with-Immediate delivers its payload directly to the address
 * named in the WRITE itself; the matching receive WR only needs to exist
 * so the HW has somewhere to place the completion (no data buffer). */
static int post_dummy_recv(struct ibv_qp *qp, uint64_t wr_id)
{
    struct ibv_recv_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id   = wr_id;
    wr.sg_list = NULL;
    wr.num_sge = 0;
    return ibv_post_recv(qp, &wr, &bad);
}

/* Defensive check: reject an RDMA op locally, with a clear message, before
 * it ever reaches the wire, instead of finding out via a cryptic
 * IBV_WC_REM_ACCESS_ERR / cascading IBV_WC_WR_FLUSH_ERR later. `base`/`size`
 * describe the registered MR window (local recvbuf/stage buffer, or the
 * peer's advertised landing buffer); `addr`/`len` is the access being
 * attempted against it. */
static int check_bounds(int rank, const char *what, uint64_t addr, uint32_t len,
                         uint64_t base, uint32_t size)
{
    if (len == 0) return 0;
    if (addr < base || len > size || addr - base > (uint64_t) size - len) {
        fprintf(stderr,
                "[rank %d] BOUNDS VIOLATION: %s addr=0x%llx len=%u outside MR "
                "[0x%llx, 0x%llx) (size=%u)\n",
                rank, what, (unsigned long long) addr, len,
                (unsigned long long) base, (unsigned long long) base + size, size);
        return -1;
    }
    return 0;
}

typedef void (*seg_done_cb)(void *user, uint64_t wr_id);

/* Waits for exactly expect_send completions on qp_next and expect_recv
 * completions on qp_prev. Because segments are posted to disjoint memory
 * offsets, cb() (the CPU reduction, for Reduce-Scatter) can run on an
 * already-arrived segment while later segments are still in flight on
 * the wire -- this is the communication/computation overlap. */
static int drain_step(pg_context_t *ctx, int expect_send, int expect_recv,
                       seg_done_cb cb, void *user)
{
    int got_send = 0, got_recv = 0;
    struct ibv_wc wc[WC_BATCH];

    while (got_send < expect_send || got_recv < expect_recv) {
        int ne = ibv_poll_cq(ctx->cq, WC_BATCH, wc);
        if (ne < 0) { fprintf(stderr, "poll_cq failed\n"); return -1; }
        for (int i = 0; i < ne; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                /* IBV_WC_WR_FLUSH_ERR just means THIS qp already entered the
                 * error state -- it is a symptom, not the cause. The real
                 * cause is whatever earlier completion (on this rank, or on
                 * the peer whose completion we can't see) first reported a
                 * non-flush status such as REM_ACCESS_ERR/REM_OP_ERR/
                 * LOC_PROT_ERR. vendor_err carries the HCA-specific detail
                 * code for that root cause. */
                const char *role =
                    (wc[i].qp_num == ctx->qp_next->qp_num) ? "qp_next(send->next)" :
                    (wc[i].qp_num == ctx->qp_prev->qp_num) ? "qp_prev(recv<-prev)" : "unknown-qp";
                fprintf(stderr,
                        "[rank %d] WC error: %s (status=%d vendor_err=0x%x qp=%u[%s] "
                        "wr_id=%llu opcode=%d byte_len=%u)\n",
                        ctx->rank, ibv_wc_status_str(wc[i].status), wc[i].status,
                        wc[i].vendor_err, wc[i].qp_num, role,
                        (unsigned long long) wc[i].wr_id, wc[i].opcode, wc[i].byte_len);
                return -1;
            }
            if (wc[i].qp_num == ctx->qp_next->qp_num) {
                got_send++;
            } else if (wc[i].qp_num == ctx->qp_prev->qp_num) {
                got_recv++;
                if (cb) cb(user, wc[i].wr_id);
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* connect_process_group() -- ring bootstrap over TCP + QP setup       */
/* ------------------------------------------------------------------ */

int connect_process_group(char *servername, void **pg_handle)
{
    (void) servername; /* ring roster comes from pg_set_topology() / CLI */

    if (g_config.world_size < 2 || g_config.my_index <= 0) {
        fprintf(stderr, "connect_process_group: topology not set "
                         "(need at least 2 processes; call pg_set_topology first)\n");
        return -1;
    }

    int N    = g_config.world_size;
    int rank = g_config.my_index - 1;
    if (rank < 0 || rank >= N) {
        fprintf(stderr, "connect_process_group: -myindex out of range\n");
        return -1;
    }

    pg_context_t *ctx = calloc(1, sizeof *ctx);
    if (!ctx) return -1;
    ctx->rank       = rank;
    ctx->world_size = N;
    ctx->next_rank  = (rank + 1) % N;
    ctx->prev_rank  = (rank - 1 + N) % N;
    ctx->ib_port    = IB_PORT_DEFAULT;
    ctx->mtu        = IBV_MTU_1024;
    ctx->sgid_idx   = -1;
    ctx->sock_next  = -1;
    ctx->sock_prev  = -1;
    ctx->eager_threshold_bytes = 0; /* always Rendezvous until autotuned */

    ctx->hostnames = calloc((size_t) N, sizeof(char *));
    for (int i = 0; i < N; i++) ctx->hostnames[i] = strdup(g_config.hostnames[i]);

    /* ---- open the IB device and build the local verbs objects ---- */
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    if (!dev_list || !dev_list[0]) {
        fprintf(stderr, "No IB devices found\n");
        return -1;
    }
    ctx->ibctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!ctx->ibctx) { fprintf(stderr, "ibv_open_device failed\n"); return -1; }

    ctx->pd = ibv_alloc_pd(ctx->ibctx);
    ctx->cq = ibv_create_cq(ctx->ibctx, CQ_SIZE, NULL, NULL, 0);
    if (!ctx->pd || !ctx->cq) { fprintf(stderr, "PD/CQ allocation failed\n"); return -1; }

    /* Two QPs per rank: one to write into "next", one that "prev" writes into. */
    ctx->qp_next = create_ring_qp(ctx);
    ctx->qp_prev = create_ring_qp(ctx);
    if (!ctx->qp_next || !ctx->qp_prev) { fprintf(stderr, "QP creation failed\n"); return -1; }

    if (modify_qp_to_init(ctx->qp_next, ctx->ib_port) ||
        modify_qp_to_init(ctx->qp_prev, ctx->ib_port)) {
        fprintf(stderr, "QP -> INIT failed\n");
        return -1;
    }

    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx->ibctx, ctx->ib_port, &port_attr)) {
        fprintf(stderr, "ibv_query_port failed\n");
        return -1;
    }
    uint16_t lid = port_attr.lid;
    union ibv_gid gid;
    memset(&gid, 0, sizeof gid);
    if (ctx->sgid_idx >= 0)
        ibv_query_gid(ctx->ibctx, ctx->ib_port, ctx->sgid_idx, &gid);

    srand48((long) (getpid() ^ time(NULL) ^ (rank << 16)));
    uint32_t my_psn_next = lrand48() & 0xffffff;
    uint32_t my_psn_prev = lrand48() & 0xffffff;

    /* ---- ring bootstrap over TCP -------------------------------------
     * Every rank does the SAME four steps; deadlock-free because
     * listen() happens before any connect() (so no connect races an
     * un-started listener for long -- tcp_connect_retry() absorbs any
     * residual startup skew) and every subsequent recv() waits on data
     * the peer sends unconditionally right after its own connect/accept,
     * never on anything that depends on our own later progress. */
    int listen_fd = tcp_listen(TCP_BASE_PORT + rank);
    if (listen_fd < 0) { fprintf(stderr, "listen on port %d failed\n", TCP_BASE_PORT + rank); return -1; }

    ctx->sock_next = tcp_connect_retry(ctx->hostnames[ctx->next_rank],
                                        TCP_BASE_PORT + ctx->next_rank, CONNECT_RETRIES);
    if (ctx->sock_next < 0) {
        fprintf(stderr, "rank %d: couldn't connect to next (%s)\n", rank, ctx->hostnames[ctx->next_rank]);
        return -1;
    }
    int one = 1;
    setsockopt(ctx->sock_next, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

    struct qp_info local_next_info = { .qpn = ctx->qp_next->qp_num, .psn = my_psn_next, .lid = lid, .gid = gid };
    if (send_all(ctx->sock_next, &local_next_info, sizeof local_next_info)) {
        fprintf(stderr, "rank %d: send qp_next info failed\n", rank);
        return -1;
    }

    struct sockaddr_in peer_addr;
    socklen_t peer_len = sizeof peer_addr;
    ctx->sock_prev = accept(listen_fd, (struct sockaddr *) &peer_addr, &peer_len);
    close(listen_fd);
    if (ctx->sock_prev < 0) { fprintf(stderr, "rank %d: accept from prev failed\n", rank); return -1; }
    setsockopt(ctx->sock_prev, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

    struct qp_info remote_prev_info;
    if (recv_all(ctx->sock_prev, &remote_prev_info, sizeof remote_prev_info)) {
        fprintf(stderr, "rank %d: recv from prev failed\n", rank);
        return -1;
    }
    if (modify_qp_to_rtr(ctx->qp_prev, ctx->ib_port, ctx->mtu, ctx->sgid_idx, &remote_prev_info) ||
        modify_qp_to_rts(ctx->qp_prev, my_psn_prev)) {
        fprintf(stderr, "rank %d: qp_prev connect failed\n", rank);
        return -1;
    }

    struct qp_info local_prev_info = { .qpn = ctx->qp_prev->qp_num, .psn = my_psn_prev, .lid = lid, .gid = gid };
    if (send_all(ctx->sock_prev, &local_prev_info, sizeof local_prev_info)) {
        fprintf(stderr, "rank %d: send qp_prev info failed\n", rank);
        return -1;
    }

    struct qp_info remote_next_info;
    if (recv_all(ctx->sock_next, &remote_next_info, sizeof remote_next_info)) {
        fprintf(stderr, "rank %d: recv qp_prev-of-next info failed\n", rank);
        return -1;
    }
    if (modify_qp_to_rtr(ctx->qp_next, ctx->ib_port, ctx->mtu, ctx->sgid_idx, &remote_next_info) ||
        modify_qp_to_rts(ctx->qp_next, my_psn_next)) {
        fprintf(stderr, "rank %d: qp_next connect failed\n", rank);
        return -1;
    }

    fprintf(stderr, "[rank %d/%d] ring connected: prev=%s next=%s "
                     "(qp_next=0x%x qp_prev=0x%x)\n",
            rank, N, ctx->hostnames[ctx->prev_rank], ctx->hostnames[ctx->next_rank],
            ctx->qp_next->qp_num, ctx->qp_prev->qp_num);

    /* Eager's stage buffer is registered ONCE, here, for the lifetime of the
     * process group -- that's the whole point of the Eager path: unlike
     * Rendezvous (which registers/exchanges a fresh landing buffer every
     * call), small messages pay zero setup cost per pg_all_reduce(). */
    ctx->eager_stage_bytes = EAGER_STAGE_BYTES;
    ctx->eager_stage = malloc(ctx->eager_stage_bytes);
    if (!ctx->eager_stage) { fprintf(stderr, "rank %d: eager stage alloc failed\n", rank); return -1; }
    ctx->eager_stage_mr = ibv_reg_mr(ctx->pd, ctx->eager_stage,
                                      ctx->eager_stage_bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->eager_stage_mr) { fprintf(stderr, "rank %d: eager stage MR reg failed\n", rank); return -1; }

    *pg_handle = ctx;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Reduce-Scatter: pipelined, CPU reduces into recvbuf as each segment  */
/* lands (staging buffer, not zero-copy -- reduction needs a landing    */
/* zone the CPU can add from without racing the caller's recvbuf).      */
/* ------------------------------------------------------------------ */

struct rs_ctx {
    void      *recvbuf;
    DATATYPE   dt;
    OPERATION  op;
    void      *stage;
    int        chunk_start_elems;
    int        nsegs;
    int        chunk_len;
};

static void rs_seg_cb(void *user, uint64_t wr_id)
{
    struct rs_ctx *rc = user;
    int seg_i = (int) wr_id;
    int seg_start, seg_len;
    seg_range(rc->chunk_len, rc->nsegs, seg_i, &seg_start, &seg_len);
    size_t esz = dt_size(rc->dt);
    void *dst = (char *) rc->recvbuf + (size_t) (rc->chunk_start_elems + seg_start) * esz;
    void *src = (char *) rc->stage + (size_t) seg_start * esz;
    reduce_apply(dst, src, seg_len, rc->dt, rc->op);
}

static int pg_reduce_scatter(pg_context_t *ctx, void *recvbuf, int count, DATATYPE dt, OPERATION op)
{
    int N = ctx->world_size, r = ctx->rank;
    size_t esz = dt_size(dt);
    if (N == 1) return 0;

    int max_chunk_elems = (count + N - 1) / N;
    size_t stage_bytes = (size_t) max_chunk_elems * esz;
    if (stage_bytes == 0) stage_bytes = esz;

    void *stage = malloc(stage_bytes);
    if (!stage) { fprintf(stderr, "stage buf alloc failed\n"); return -1; }
    struct ibv_mr *stage_mr = ibv_reg_mr(ctx->pd, stage, stage_bytes,
                                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    struct ibv_mr *recv_mr  = ibv_reg_mr(ctx->pd, recvbuf, (size_t) count * esz,
                                         IBV_ACCESS_LOCAL_WRITE);
    if (!stage_mr || !recv_mr) { fprintf(stderr, "MR registration failed (reduce_scatter)\n"); return -1; }

    struct buf_info my_landing = { .addr = (uint64_t) stage, .rkey = stage_mr->rkey, .size = (uint32_t) stage_bytes };
    struct buf_info next_landing;
    if (exchange_landing_info(ctx, &my_landing, &next_landing)) {
        fprintf(stderr, "landing-info exchange failed (reduce_scatter)\n");
        return -1;
    }

    /* nsegs_out captured from the PREVIOUS iteration: gates whether next's
     * ack (below) is even expected this step -- there's only something to
     * protect if we actually wrote into next's stage buffer last time. */
    int prev_nsegs_out = 0;

    for (int s = 0; s < N - 1; s++) {
        int send_idx = ((r - s) % N + N) % N;
        int recv_idx = ((r - s - 1) % N + N) % N;
        int s_start, s_len, r_start, r_len;
        chunk_range(count, N, send_idx, &s_start, &s_len);
        chunk_range(count, N, recv_idx, &r_start, &r_len);

        int nsegs_out = s_len > 0 ? (s_len < PIPE_SEGS ? s_len : PIPE_SEGS) : 0;
        int nsegs_in  = r_len > 0 ? (r_len < PIPE_SEGS ? r_len : PIPE_SEGS) : 0;

        /* The stage buffer is a single, reused region (same address every
         * step). An RDMA-write send-completion only proves the bytes landed
         * in next's memory -- it says nothing about whether next's CPU has
         * finished reduce_apply()-ing the *previous* step's contents out of
         * that same buffer yet. Without an explicit ack, a fast sender can
         * race ahead and overwrite data its peer hasn't consumed.
         *
         * Gate this on PREV_NSEGS_OUT (what *this* rank sent last step),
         * not the current step's count: the ack we're waiting for is next's
         * confirmation that it drained *that* write, and next only sends
         * one when it actually received something last step. Getting this
         * gate wrong desyncs the ack byte-stream against the very next
         * landing-info exchange (All-Gather's) -- which is exactly what
         * produced a corrupted remote addr/rkey and the resulting
         * IBV_WC_REM_ACCESS_ERR / cascading IBV_WC_WR_FLUSH_ERR before. */
        if (s > 0 && prev_nsegs_out > 0) {
            char ack;
            if (recv_all(ctx->sock_next, &ack, 1)) {
                fprintf(stderr, "reduce_scatter: ack-from-next failed at step %d\n", s);
                return -1;
            }
        }

        /* Pre-post receive placeholders before the peer starts writing. */
        for (int i = 0; i < nsegs_in; i++) post_dummy_recv(ctx->qp_prev, (uint64_t) i);

        /* Post every outbound segment up front: sends of later segments
         * proceed on the wire while earlier ones are still completing --
         * this, combined with reducing arrived segments immediately
         * below (via drain_step's callback), is the comm/compute overlap. */
        for (int i = 0; i < nsegs_out; i++) {
            int seg_start, seg_len;
            seg_range(s_len, nsegs_out, i, &seg_start, &seg_len);
            uint64_t laddr = (uint64_t) ((char *) recvbuf + (size_t) (s_start + seg_start) * esz);
            uint64_t raddr = next_landing.addr + (size_t) seg_start * esz;
            uint32_t wlen  = (uint32_t) (seg_len * esz);

            if (check_bounds(ctx->rank, "reduce_scatter local(send-src)",
                              laddr, wlen, (uint64_t) recvbuf, (uint32_t) (count * esz)) ||
                check_bounds(ctx->rank, "reduce_scatter remote(next stage)",
                              raddr, wlen, next_landing.addr, next_landing.size)) {
                return -1;
            }

            if (post_write_imm(ctx->qp_next, laddr, recv_mr->lkey, wlen,
                                raddr, next_landing.rkey, (uint32_t) ((s << 8) | i), (uint64_t) i)) {
                fprintf(stderr, "post_write_imm failed (reduce_scatter step %d)\n", s);
                return -1;
            }
        }

        struct rs_ctx rc = { .recvbuf = recvbuf, .dt = dt, .op = op, .stage = stage,
                              .chunk_start_elems = r_start, .nsegs = nsegs_in, .chunk_len = r_len };
        if (drain_step(ctx, nsegs_out, nsegs_in, rs_seg_cb, &rc)) return -1;

        /* I've now fully reduce_apply()'d every segment prev wrote into my
         * stage buffer this step -- tell prev it may safely reuse that same
         * buffer for step s+1's writes. Skip on the LAST step: there is no
         * step N-1, so nobody will ever wait for this one -- sending it
         * anyway is exactly the extra, unconsumed byte that caused the
         * desync described above. */
        if (s < N - 2 && nsegs_in > 0) {
            char ack = 1;
            if (send_all(ctx->sock_prev, &ack, 1)) {
                fprintf(stderr, "reduce_scatter: ack-to-prev failed at step %d\n", s);
                return -1;
            }
        }

        prev_nsegs_out = nsegs_out;
    }

    ibv_dereg_mr(stage_mr);
    ibv_dereg_mr(recv_mr);
    free(stage);
    return 0;
}

/* ------------------------------------------------------------------ */
/* All-Gather: zero-copy. Every segment is RDMA-Written straight from   */
/* the sender's recvbuf chunk into the SAME byte offset of the peer's   */
/* recvbuf -- no staging buffer, no memcpy on either side.              */
/* ------------------------------------------------------------------ */

static int pg_all_gather(pg_context_t *ctx, void *recvbuf, int count, DATATYPE dt)
{
    int N = ctx->world_size, r = ctx->rank;
    size_t esz = dt_size(dt);
    if (N == 1) return 0;

    struct ibv_mr *recv_mr = ibv_reg_mr(ctx->pd, recvbuf, (size_t) count * esz,
                                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!recv_mr) { fprintf(stderr, "recvbuf MR registration failed (all_gather)\n"); return -1; }

    struct buf_info my_landing = { .addr = (uint64_t) recvbuf, .rkey = recv_mr->rkey,
                                    .size = (uint32_t) (count * esz) };
    struct buf_info next_landing;
    if (exchange_landing_info(ctx, &my_landing, &next_landing)) {
        fprintf(stderr, "landing-info exchange failed (all_gather)\n");
        return -1;
    }

    for (int s = 0; s < N - 1; s++) {
        int send_idx = ((r - s + 1) % N + N) % N;
        int recv_idx = ((r - s) % N + N) % N;
        int s_start, s_len, r_start, r_len;
        chunk_range(count, N, send_idx, &s_start, &s_len);
        chunk_range(count, N, recv_idx, &r_start, &r_len);

        int nsegs_out = s_len > 0 ? (s_len < PIPE_SEGS ? s_len : PIPE_SEGS) : 0;
        int nsegs_in  = r_len > 0 ? (r_len < PIPE_SEGS ? r_len : PIPE_SEGS) : 0;

        for (int i = 0; i < nsegs_in; i++) post_dummy_recv(ctx->qp_prev, (uint64_t) i);

        for (int i = 0; i < nsegs_out; i++) {
            int seg_start, seg_len;
            seg_range(s_len, nsegs_out, i, &seg_start, &seg_len);
            uint64_t laddr = (uint64_t) ((char *) recvbuf + (size_t) (s_start + seg_start) * esz);
            /* zero-copy: target offset is the SAME chunk position in the
             * peer's own recvbuf -- final data lands there directly. */
            uint64_t raddr = next_landing.addr + (size_t) (s_start + seg_start) * esz;
            uint32_t wlen  = (uint32_t) (seg_len * esz);

            if (check_bounds(ctx->rank, "all_gather local(send-src)",
                              laddr, wlen, (uint64_t) recvbuf, (uint32_t) (count * esz)) ||
                check_bounds(ctx->rank, "all_gather remote(next recvbuf)",
                              raddr, wlen, next_landing.addr, next_landing.size)) {
                return -1;
            }

            if (post_write_imm(ctx->qp_next, laddr, recv_mr->lkey, wlen,
                                raddr, next_landing.rkey, (uint32_t) ((s << 8) | i), (uint64_t) i)) {
                fprintf(stderr, "post_write_imm failed (all_gather step %d)\n", s);
                return -1;
            }
        }

        /* No callback needed: the RDMA Write already deposited the final
         * bytes; the completion is purely a delivery signal. */
        if (drain_step(ctx, nsegs_out, nsegs_in, NULL, NULL)) return -1;
    }

    ibv_dereg_mr(recv_mr);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Eager protocol: two-sided Send/Recv, no per-call TCP handshake.      */
/*                                                                      */
/* Rendezvous (above) pays, on every single call: an ibv_reg_mr() of the*/
/* caller's buffer plus a TCP round trip (exchange_landing_info) for    */
/* EACH of Reduce-Scatter and All-Gather, before it may issue a single  */
/* RDMA Write -- because a one-sided Write needs the target's (addr,    */
/* rkey) in hand before it can be posted. For a small message that      */
/* metadata exchange costs far more than moving the actual bytes.       */
/*                                                                      */
/* Eager sidesteps all of that by using ordinary two-sided IBV_WR_SEND  */
/* / IBV_WR_RECV over the SAME persistent ring QPs:                     */
/*   - The destination address for a two-sided message is decided by   */
/*     the RECEIVER, via wherever it points its own posted recv WR's    */
/*     SGE -- not negotiated with the sender. Since both sides already  */
/*     compute the same chunk/segment layout from (count, N, step) with */
/*     no coordination (see chunk_range/seg_range), the receiver can    */
/*     always pre-post a recv WR at exactly the right local address.    */
/*     That means NO landing-info exchange is needed at all, for        */
/*     either phase.                                                    */
/*   - Reduce-Scatter still needs a landing zone the CPU can safely add */
/*     from (same reason as Rendezvous), but instead of a fresh MR      */
/*     registered per call, it reuses ctx->eager_stage -- registered    */
/*     ONCE in connect_process_group(), for the process group's whole   */
/*     lifetime.                                                        */
/*   - All-Gather posts its recv WR straight into the destination slot  */
/*     of recvbuf itself, so arriving data lands in its final position  */
/*     directly -- no separate staging copy.                            */
/*   - Rendezvous needed an explicit software ack to stop a sender from */
/*     overwriting a reused stage buffer before its peer finished       */
/*     draining it (see pg_reduce_scatter). Eager needs none: a SEND    */
/*     cannot be delivered until the receiver has posted a matching     */
/*     RECV WR, and this code only posts step s+1's RECV after fully    */
/*     draining step s -- so the receiver's own pacing IS the flow      */
/*     control, for free, courtesy of two-sided send/recv semantics.    */
/* ------------------------------------------------------------------ */

static int post_send_data(struct ibv_qp *qp, uint64_t laddr, uint32_t lkey,
                           uint32_t len, uint64_t wr_id)
{
    struct ibv_sge sge = { .addr = laddr, .length = len, .lkey = lkey };
    struct ibv_send_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id      = wr_id;
    wr.opcode     = IBV_WR_SEND;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    return ibv_post_send(qp, &wr, &bad);
}

static int post_recv_into(struct ibv_qp *qp, uint64_t laddr, uint32_t lkey,
                           uint32_t len, uint64_t wr_id)
{
    struct ibv_sge sge = { .addr = laddr, .length = len, .lkey = lkey };
    struct ibv_recv_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id   = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    return ibv_post_recv(qp, &wr, &bad);
}

/* Registration cache keyed on pointer identity: a benchmark loop or a
 * training loop calling pg_all_reduce() on the SAME sendbuf/recvbuf every
 * iteration should only pay ibv_reg_mr() once, not once per call. */
static struct ibv_mr *get_cached_mr(pg_context_t *ctx, void *buf, size_t bytes)
{
    if (ctx->eager_mr && ctx->eager_mr_ptr == buf && ctx->eager_mr_size >= bytes)
        return ctx->eager_mr;

    if (ctx->eager_mr) { ibv_dereg_mr(ctx->eager_mr); ctx->eager_mr = NULL; }

    ctx->eager_mr = ibv_reg_mr(ctx->pd, buf, bytes, IBV_ACCESS_LOCAL_WRITE);
    ctx->eager_mr_ptr  = buf;
    ctx->eager_mr_size = bytes;
    return ctx->eager_mr;
}

static int pg_reduce_scatter_eager(pg_context_t *ctx, void *recvbuf, int count,
                                    DATATYPE dt, OPERATION op)
{
    int N = ctx->world_size, r = ctx->rank;
    size_t esz = dt_size(dt);
    if (N == 1) return 0;

    size_t total_bytes = (size_t) count * esz;
    struct ibv_mr *recv_mr = get_cached_mr(ctx, recvbuf, total_bytes);
    if (!recv_mr) { fprintf(stderr, "rank %d: eager recvbuf MR reg failed\n", r); return -1; }

    int max_chunk_elems = (count + N - 1) / N;
    size_t need_stage = (size_t) max_chunk_elems * esz;
    if (need_stage > ctx->eager_stage_bytes) {
        fprintf(stderr, "rank %d: message too large for eager stage buffer "
                         "(chunk=%zu B > %zu B) -- route through Rendezvous instead\n",
                r, need_stage, ctx->eager_stage_bytes);
        return -1;
    }

    for (int s = 0; s < N - 1; s++) {
        int send_idx = ((r - s) % N + N) % N;
        int recv_idx = ((r - s - 1) % N + N) % N;
        int s_start, s_len, r_start, r_len;
        chunk_range(count, N, send_idx, &s_start, &s_len);
        chunk_range(count, N, recv_idx, &r_start, &r_len);

        int nsegs_out = s_len > 0 ? (s_len < PIPE_SEGS ? s_len : PIPE_SEGS) : 0;
        int nsegs_in  = r_len > 0 ? (r_len < PIPE_SEGS ? r_len : PIPE_SEGS) : 0;

        /* No ack needed here (see comment block above): posting this
         * step's recv WRs only NOW is what makes reuse of eager_stage
         * safe -- prev's matching sends simply can't land until we do. */
        for (int i = 0; i < nsegs_in; i++) {
            int seg_start, seg_len;
            seg_range(r_len, nsegs_in, i, &seg_start, &seg_len);
            uint64_t addr = (uint64_t) ((char *) ctx->eager_stage + (size_t) seg_start * esz);
            post_recv_into(ctx->qp_prev, addr, ctx->eager_stage_mr->lkey,
                            (uint32_t) (seg_len * esz), (uint64_t) i);
        }

        for (int i = 0; i < nsegs_out; i++) {
            int seg_start, seg_len;
            seg_range(s_len, nsegs_out, i, &seg_start, &seg_len);
            uint64_t laddr = (uint64_t) ((char *) recvbuf + (size_t) (s_start + seg_start) * esz);
            uint32_t wlen  = (uint32_t) (seg_len * esz);
            if (check_bounds(r, "eager reduce_scatter local(send-src)",
                              laddr, wlen, (uint64_t) recvbuf, (uint32_t) total_bytes)) {
                return -1;
            }
            if (post_send_data(ctx->qp_next, laddr, recv_mr->lkey, wlen, (uint64_t) i)) {
                fprintf(stderr, "rank %d: eager post_send failed (reduce_scatter step %d)\n", r, s);
                return -1;
            }
        }

        /* Same rs_ctx/rs_seg_cb as Rendezvous -- reduction doesn't care
         * whether the data arrived via RDMA Write or two-sided Send. */
        struct rs_ctx rc = { .recvbuf = recvbuf, .dt = dt, .op = op, .stage = ctx->eager_stage,
                              .chunk_start_elems = r_start, .nsegs = nsegs_in, .chunk_len = r_len };
        if (drain_step(ctx, nsegs_out, nsegs_in, rs_seg_cb, &rc)) return -1;
    }
    return 0;
}

static int pg_all_gather_eager(pg_context_t *ctx, void *recvbuf, int count, DATATYPE dt)
{
    int N = ctx->world_size, r = ctx->rank;
    size_t esz = dt_size(dt);
    if (N == 1) return 0;

    size_t total_bytes = (size_t) count * esz;
    struct ibv_mr *recv_mr = get_cached_mr(ctx, recvbuf, total_bytes);
    if (!recv_mr) { fprintf(stderr, "rank %d: eager recvbuf MR reg failed (all_gather)\n", r); return -1; }

    for (int s = 0; s < N - 1; s++) {
        int send_idx = ((r - s + 1) % N + N) % N;
        int recv_idx = ((r - s) % N + N) % N;
        int s_start, s_len, r_start, r_len;
        chunk_range(count, N, send_idx, &s_start, &s_len);
        chunk_range(count, N, recv_idx, &r_start, &r_len);

        int nsegs_out = s_len > 0 ? (s_len < PIPE_SEGS ? s_len : PIPE_SEGS) : 0;
        int nsegs_in  = r_len > 0 ? (r_len < PIPE_SEGS ? r_len : PIPE_SEGS) : 0;

        /* Zero-copy via placement, not via RDMA: the recv WR's own SGE
         * points straight at this step's final slot inside recvbuf, so
         * the arriving segment needs no separate staging or memcpy. */
        for (int i = 0; i < nsegs_in; i++) {
            int seg_start, seg_len;
            seg_range(r_len, nsegs_in, i, &seg_start, &seg_len);
            uint64_t addr = (uint64_t) ((char *) recvbuf + (size_t) (r_start + seg_start) * esz);
            post_recv_into(ctx->qp_prev, addr, recv_mr->lkey,
                            (uint32_t) (seg_len * esz), (uint64_t) i);
        }

        for (int i = 0; i < nsegs_out; i++) {
            int seg_start, seg_len;
            seg_range(s_len, nsegs_out, i, &seg_start, &seg_len);
            uint64_t laddr = (uint64_t) ((char *) recvbuf + (size_t) (s_start + seg_start) * esz);
            uint32_t wlen  = (uint32_t) (seg_len * esz);
            if (check_bounds(r, "eager all_gather local(send-src)",
                              laddr, wlen, (uint64_t) recvbuf, (uint32_t) total_bytes)) {
                return -1;
            }
            if (post_send_data(ctx->qp_next, laddr, recv_mr->lkey, wlen, (uint64_t) i)) {
                fprintf(stderr, "rank %d: eager post_send failed (all_gather step %d)\n", r, s);
                return -1;
            }
        }

        if (drain_step(ctx, nsegs_out, nsegs_in, NULL, NULL)) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* pg_all_reduce: protocol implementations + size-based router          */
/* ------------------------------------------------------------------ */

/* Strictly in-place: no memcpy. sendbuf/recvbuf must be the same pointer --
 * reduce-scatter and all-gather both operate directly on that one buffer. */
static int pg_all_reduce_rendezvous(pg_context_t *ctx, void *sendbuf, void *recvbuf,
                                     int count, DATATYPE dt, OPERATION op)
{
    if (sendbuf != recvbuf) {
        fprintf(stderr, "pg_all_reduce: library is strictly in-place; sendbuf must equal recvbuf\n");
        return -1;
    }
    if (ctx->world_size == 1) return 0;
    if (pg_reduce_scatter(ctx, recvbuf, count, dt, op)) return -1;
    if (pg_all_gather(ctx, recvbuf, count, dt)) return -1;
    return 0;
}

static int pg_all_reduce_eager(pg_context_t *ctx, void *sendbuf, void *recvbuf,
                                int count, DATATYPE dt, OPERATION op)
{
    if (sendbuf != recvbuf) {
        fprintf(stderr, "pg_all_reduce: library is strictly in-place; sendbuf must equal recvbuf\n");
        return -1;
    }
    if (ctx->world_size == 1) return 0;
    if (pg_reduce_scatter_eager(ctx, recvbuf, count, dt, op)) return -1;
    if (pg_all_gather_eager(ctx, recvbuf, count, dt)) return -1;
    return 0;
}

/* Simple linear (non-wrapping) ring broadcast of a single value from rank
 * 0 to every other rank, reusing the persistent control sockets: rank 0
 * sends to next; ranks 1..N-2 receive from prev then relay to next; rank
 * N-1 receives from prev and stops (relaying further would loop back to
 * rank 0, which isn't expecting anything). Used only to make sure every
 * rank agrees on the SAME autotuned threshold (see below). */
static int broadcast_from_rank0(pg_context_t *ctx, size_t *value)
{
    int N = ctx->world_size, r = ctx->rank;
    if (N == 1) return 0;
    if (r == 0) {
        return send_all(ctx->sock_next, value, sizeof *value);
    }
    if (recv_all(ctx->sock_prev, value, sizeof *value)) return -1;
    if (r != N - 1) return send_all(ctx->sock_next, value, sizeof *value);
    return 0;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* Runs AUTOTUNE_WARMUP untimed + AUTOTUNE_TRIALS timed all-reduces of the
 * requested protocol and returns the average time per call, in ms. Every
 * call here is a genuine ring collective, so all ranks stay in lockstep
 * automatically as long as they all call this with the same (size,
 * protocol) sequence -- which pg_autotune_threshold() guarantees by never
 * branching on a per-rank measurement mid-sweep. */
static double time_protocol_ms(pg_context_t *ctx, void *buf, int count, int use_eager)
{
    for (int i = 0; i < AUTOTUNE_WARMUP; i++) {
        memset(buf, 0, (size_t) count * sizeof(float));
        if (use_eager) pg_all_reduce_eager(ctx, buf, buf, count, DT_FLOAT, OP_SUM);
        else            pg_all_reduce_rendezvous(ctx, buf, buf, count, DT_FLOAT, OP_SUM);
    }
    double t0 = now_ms();
    for (int i = 0; i < AUTOTUNE_TRIALS; i++) {
        memset(buf, 0, (size_t) count * sizeof(float));
        if (use_eager) pg_all_reduce_eager(ctx, buf, buf, count, DT_FLOAT, OP_SUM);
        else            pg_all_reduce_rendezvous(ctx, buf, buf, count, DT_FLOAT, OP_SUM);
    }
    double t1 = now_ms();
    return (t1 - t0) / AUTOTUNE_TRIALS;
}

/* Sweeps message sizes from AUTOTUNE_MIN_BYTES to AUTOTUNE_MAX_BYTES
 * (doubling each step), times BOTH protocols at every size, and records
 * the smallest size at which Rendezvous first becomes faster than Eager --
 * that's the crossover threshold pg_all_reduce()'s router uses from then
 * on. Every rank runs the identical, hardcoded size sequence (never
 * short-circuiting on its own local timing) so the sweep's calls stay
 * synchronized ring-wide; only the FINAL verdict can differ slightly
 * between ranks due to timing jitter, so rank 0's verdict is broadcast to
 * everyone at the end to guarantee they all route identically -- if two
 * ranks routed a given call to different protocols, the ring would
 * deadlock (one side posting RDMA Writes, the other expecting Sends). */
int pg_autotune_threshold(void *pg_handle)
{
    pg_context_t *ctx = pg_handle;
    if (!ctx) return -1;

    size_t max_bytes = AUTOTUNE_MAX_BYTES;
    void *buf = malloc(max_bytes);
    if (!buf) {
        fprintf(stderr, "rank %d: autotune alloc failed\n", ctx->rank);
        return -1;
    }

    if (ctx->rank == 0)
        fprintf(stderr, "[rank 0] auto-tuning eager/rendezvous crossover "
                         "(%d..%zu bytes)...\n", AUTOTUNE_MIN_BYTES, max_bytes);

    size_t threshold = max_bytes + 1;  /* default: eager wins everywhere tested */
    int crossover_found = 0;

    for (size_t bytes = AUTOTUNE_MIN_BYTES; bytes <= max_bytes; bytes *= 2) {
        int count = (int) (bytes / sizeof(float));
        if (count < 1) count = 1;

        double t_eager = time_protocol_ms(ctx, buf, count, 1);
        double t_rend  = time_protocol_ms(ctx, buf, count, 0);

        if (ctx->rank == 0) {
            fprintf(stderr, "  size=%8zu B  eager=%9.4f ms  rendezvous=%9.4f ms  -> %s\n",
                    bytes, t_eager, t_rend, (t_rend < t_eager) ? "rendezvous" : "eager");
        }

        /* Keep sweeping the full range on every rank regardless (no early
         * break): all ranks must issue the exact same call sequence for
         * the collectives above to stay synchronized. Only remember the
         * FIRST crossover we see. */
        if (!crossover_found && t_rend < t_eager) {
            threshold = bytes;
            crossover_found = 1;
        }
    }

    free(buf);

    size_t agreed = threshold;
    if (broadcast_from_rank0(ctx, &agreed)) {
        fprintf(stderr, "rank %d: threshold broadcast failed\n", ctx->rank);
        return -1;
    }

    ctx->eager_threshold_bytes = agreed;
    fprintf(stderr, "[rank %d] eager/rendezvous crossover threshold = %zu bytes\n",
            ctx->rank, ctx->eager_threshold_bytes);
    return 0;
}

/* Router: dispatches to Eager for small messages, Rendezvous otherwise.
 * The crossover point is whatever pg_autotune_threshold() last measured
 * (0, i.e. "always Rendezvous", if it hasn't been called). */
int pg_all_reduce(void *sendbuf, void *recvbuf, int count,
                   DATATYPE datatype, OPERATION op, void *pg_handle)
{
    pg_context_t *ctx = pg_handle;
    if (!ctx) return -1;
    size_t esz = dt_size(datatype);
    if (esz == 0 || count <= 0) { fprintf(stderr, "pg_all_reduce: bad count/datatype\n"); return -1; }

    size_t nbytes = (size_t) count * esz;
    if (nbytes < ctx->eager_threshold_bytes)
        return pg_all_reduce_eager(ctx, sendbuf, recvbuf, count, datatype, op);
    return pg_all_reduce_rendezvous(ctx, sendbuf, recvbuf, count, datatype, op);
}

int pg_close(void *pg_handle)
{
    pg_context_t *ctx = pg_handle;
    if (!ctx) return -1;
    int rc = 0;

    if (ctx->eager_mr && ibv_dereg_mr(ctx->eager_mr))            { fprintf(stderr, "dereg eager_mr failed\n"); rc = -1; }
    if (ctx->eager_stage_mr && ibv_dereg_mr(ctx->eager_stage_mr)) { fprintf(stderr, "dereg eager_stage_mr failed\n"); rc = -1; }
    free(ctx->eager_stage);

    if (ctx->qp_next && ibv_destroy_qp(ctx->qp_next)) { fprintf(stderr, "destroy qp_next failed\n"); rc = -1; }
    if (ctx->qp_prev && ibv_destroy_qp(ctx->qp_prev)) { fprintf(stderr, "destroy qp_prev failed\n"); rc = -1; }
    if (ctx->cq && ibv_destroy_cq(ctx->cq))            { fprintf(stderr, "destroy cq failed\n"); rc = -1; }
    if (ctx->pd && ibv_dealloc_pd(ctx->pd))            { fprintf(stderr, "dealloc pd failed\n"); rc = -1; }
    if (ctx->ibctx && ibv_close_device(ctx->ibctx))    { fprintf(stderr, "close device failed\n"); rc = -1; }

    if (ctx->sock_next >= 0) close(ctx->sock_next);
    if (ctx->sock_prev >= 0) close(ctx->sock_prev);

    if (ctx->hostnames) {
        for (int i = 0; i < ctx->world_size; i++) free(ctx->hostnames[i]);
        free(ctx->hostnames);
    }
    free(ctx);
    return rc;
}

/* ------------------------------------------------------------------ */
/* CLI / driver                                                         */
/* ------------------------------------------------------------------ */

/* ==================== TEMP: -tune-pipe (delete when done) ==================== */
/* For each message size, times `iters` pg_all_reduce() calls for each
 * candidate PIPE_SEGS value. Every rank walks the same lists in the same
 * order, so the whole ring always uses the same segment count. Max is 64:
 * each step posts up to PIPE_SEGS writes/recvs (SQ_DEPTH/RQ_DEPTH = 64) and
 * the segment index travels in 8 bits of the immediate. Sizes start at
 * 16 KB: below that the chunks are so small that segmenting barely matters. */
static int tune_pipe_segs(void *pg_handle, int rank, int nhosts, int iters)
{
    pg_context_t *ctx = pg_handle;
    static const int    segs[]  = { 4, 8, 16, 32, 64 };
    static const size_t sizes[] = { 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024 };
    const int nsegs  = (int) (sizeof segs / sizeof segs[0]);
    const int nsizes = (int) (sizeof sizes / sizeof sizes[0]);
    const size_t max_bytes = sizes[nsizes - 1];

    float *recvbuf = malloc(max_bytes);
    if (!recvbuf) { fprintf(stderr, "alloc failed\n"); return -1; }

    double expected = 0.0;
    for (int p = 0; p < nhosts; p++) expected += (p + 1);

    if (rank == 0) {
        printf("\n[tune-pipe] total ms for %d x pg_all_reduce (* = fastest for that size)\n", iters);
        printf("%10s %-10s", "size [B]", "router");
        for (int k = 0; k < nsegs; k++) printf("   segs=%-6d", segs[k]);
        printf("\n");
    }

    int saved = g_pipe_segs, rc = 0;
    for (int s = 0; s < nsizes && rc == 0; s++) {
        int count = (int) (sizes[s] / sizeof(float));
        double t[sizeof segs / sizeof segs[0]];
        int ok = 1, best = 0;

        for (int k = 0; k < nsegs; k++) {
            g_pipe_segs = segs[k];

            for (int w = 0; w < WARMUP_ITERS; w++) {
                for (int i = 0; i < count; i++) recvbuf[i] = (float) (rank + 1);
                pg_all_reduce(recvbuf, recvbuf, count, DT_FLOAT, OP_SUM, pg_handle);
            }

            double t0 = now_ms();
            for (int it = 0; it < iters; it++) {
                for (int i = 0; i < count; i++) recvbuf[i] = (float) (rank + 1);
                if (pg_all_reduce(recvbuf, recvbuf, count, DT_FLOAT, OP_SUM, pg_handle)) {
                    fprintf(stderr, "pg_all_reduce failed (size=%zu, PIPE_SEGS=%d)\n",
                            sizes[s], g_pipe_segs);
                    rc = -1;
                    break;
                }
            }
            if (rc) break;
            t[k] = now_ms() - t0;
            if (t[k] < t[best]) best = k;

            for (int i = 0; i < count; i++)
                if (fabs((double) recvbuf[i] - expected) > 1e-3) { ok = 0; break; }
        }

        if (rank == 0 && rc == 0) {
            printf("%10zu %-10s", sizes[s], sizes[s] < ctx->eager_threshold_bytes ? "eager" : "rendezvous");
            for (int k = 0; k < nsegs; k++) printf(" %11.3f%c ", t[k], k == best ? '*' : ' ');
            printf("%s\n", ok ? "" : " MISMATCH");
            fflush(stdout);
        }
    }
    g_pipe_segs = saved;

    free(recvbuf);
    return rc;
}
/* ================== end TEMP: -tune-pipe ================== */

static void usage(const char *argv0)
{
    printf("Usage:\n");
    printf("  %s -myindex <1-based index> -list <host1> <host2> ... [-n iters]\n", argv0);
    printf("\nExample (4 processes):\n");
    printf("  <mlxstud01> %s -myindex 1 -list mlxstud01 mlxstud02 mlxstud03 mlxstud04\n", argv0);
}

int main(int argc, char *argv[])
{
    int   my_index = -1;
    char *hostlist[MAX_PROCESSES];
    int   nhosts = 0;
    int   iters  = BENCH_ITERS;
    int   tune_pipe = 0;   /* TEMP: -tune-pipe */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-myindex") && i + 1 < argc) {
            my_index = (int) strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-list")) {
            i++;
            while (i < argc && argv[i][0] != '-') {
                if (nhosts >= MAX_PROCESSES) { fprintf(stderr, "too many hosts\n"); return 1; }
                hostlist[nhosts++] = argv[i++];
            }
            i--; /* outer loop will re-increment */
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            iters = (int) strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-tune-pipe")) {   /* TEMP */
            tune_pipe = 1;
        } else {
            fprintf(stderr, "Unrecognized option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (my_index < 1 || nhosts < 2 || my_index > nhosts || iters < 1) {
        usage(argv[0]);
        return 1;
    }

    pg_set_topology(my_index, hostlist, nhosts);

    void *pg_handle = NULL;
    if (connect_process_group(NULL, &pg_handle)) {
        fprintf(stderr, "connect_process_group failed\n");
        return 1;
    }

    /* Auto-tune the Eager/Rendezvous crossover point once, right after
     * connecting, before any timed work: pg_all_reduce() uses the result
     * to route every subsequent call by message size. */
    if (pg_autotune_threshold(pg_handle)) {
        fprintf(stderr, "pg_autotune_threshold failed\n");
        return 1;
    }

    /* TEMP: -tune-pipe runs the PIPE_SEGS sweep instead of the benchmark */
    if (tune_pipe) {
        int rc = tune_pipe_segs(pg_handle, my_index - 1, nhosts, iters);
        pg_close(pg_handle);
        return rc ? 1 : 0;
    }

    /* Interesting sizes: a single float (pure latency), a small message
     * (Eager territory), mid-size, a large message (Rendezvous / bandwidth),
     * and the largest size the Eager stage buffer was sized for. */
    static const size_t sizes[] = { 4, 1024, 16 * 1024, 1024 * 1024, 4 * 1024 * 1024 };
    const int nsizes = (int) (sizeof sizes / sizeof sizes[0]);
    const size_t max_bytes = sizes[nsizes - 1];

    int rank = my_index - 1;
    float *recvbuf = malloc(max_bytes);
    if (!recvbuf) { fprintf(stderr, "alloc failed\n"); return 1; }

    double expected = 0.0;
    for (int p = 0; p < nhosts; p++) expected += (p + 1);

    for (int s = 0; s < nsizes; s++) {
        int count = (int) (sizes[s] / sizeof(float));

        /* Warmup: a few untimed calls so the timed loop doesn't pay one-time
         * costs (first-touch page faults, MR registration, cold NIC paths). */
        for (int w = 0; w < WARMUP_ITERS; w++) {
            for (int i = 0; i < count; i++) recvbuf[i] = (float) (rank + 1);
            pg_all_reduce(recvbuf, recvbuf, count, DT_FLOAT, OP_SUM, pg_handle);
        }

        double t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < count; i++) recvbuf[i] = (float) (rank + 1);
            if (pg_all_reduce(recvbuf, recvbuf, count, DT_FLOAT, OP_SUM, pg_handle)) {
                fprintf(stderr, "pg_all_reduce failed\n");
                return 1;
            }
        }
        double total_ms = now_ms() - t0;

        /* Verify the last result (outside the timer) */
        int ok = 1;
        for (int i = 0; i < count; i++) {
            if (fabs((double) recvbuf[i] - expected) > 1e-3) { ok = 0; break; }
        }

        printf("[rank %d] size=%8zu B: %d x all_reduce took %.3f ms total (%.4f ms avg) -> %s\n",
               rank, sizes[s], iters, total_ms, total_ms / iters, ok ? "OK" : "MISMATCH");
    }

    free(recvbuf);
    pg_close(pg_handle);
    return 0;
}
