# Ring All-Reduce over RDMA

Exercise 3 (Low-Latency Protocols) of the Communication Networks course. A
single-threaded C application that connects 2 to 4 processes in a ring using the
Verbs API and implements Reduce-Scatter, All-Gather and All-Reduce. It runs the
same collective over an **Eager** and a **Rendezvous** protocol and measures both,
so the two can be compared.

The exercise description is in `Exercise 3 - Ring All-Reduce - RDMA.pdf`. All the
code is in `main.c`.

## Requirements

- Linux hosts with an InfiniBand device
- `libibverbs` development headers (for example `libibverbs-dev`)
- GCC or Clang, and CMake 3.10 or later

The project does not build on Windows, because there is no Verbs implementation
there.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces `build/HCP-project`. Use a Release build when measuring, since CMake
adds no optimization flags by default.

To build without CMake:

```sh
gcc -O2 -std=gnu11 main.c -o test -libverbs -lm
```

## Run

Start one process on each host, with the same host list everywhere:

```sh
<mlxstud01> ./test -myindex 1 -list mlxstud01 mlxstud02 mlxstud03 mlxstud04
<mlxstud02> ./test -myindex 2 -list mlxstud01 mlxstud02 mlxstud03 mlxstud04
<mlxstud03> ./test -myindex 3 -list mlxstud01 mlxstud02 mlxstud03 mlxstud04
<mlxstud04> ./test -myindex 4 -list mlxstud01 mlxstud02 mlxstud03 mlxstud04
```

| Option | Meaning |
| --- | --- |
| `-myindex <i>` | This process's 1-based position in the list |
| `-list <host1> <host2> ...` | The ring members, 2 to 4 hosts, in ring order |
| `-n <iters>` | Timed All-Reduce calls per size and protocol (default 1000) |

Each rank retries its connection to the next rank for about 20 seconds, so all
processes should be started within that window. Rank `r` listens on TCP port
`18800 + r`, and these ports must be free and reachable between the hosts.

## Output

The program uses `float` / `OP_SUM`, with each rank contributing `rank + 1` in every
element. It tests 10 message sizes, from 16 B to 4 MB (multiplying by 4 each time).
For each size it runs both protocols, alternating which one goes first so that
neither always benefits from running second. Each protocol gets 5 untimed warmup
calls, then the timed calls.

Each rank prints one line per size and protocol:

```text
[rank <r>] protocol=<Eager|Rendezvous> size=<bytes> B: <iters> x all_reduce took <total> ms total (<avg> ms avg) -> OK
```

The line ends in `OK` if the result of the final call matches the expected sum and
`sendbuf` was left unchanged, and in `MISMATCH` otherwise.

## Design

### Ring setup

`connect_process_group()` creates two RC Queue Pairs per rank: `qp_next` and
`qp_prev`. TCP is used only for bootstrapping, to exchange QP numbers, PSNs, LIDs
and buffer descriptors. The sockets are closed before any collective runs, so all
communication after that goes over RDMA.

### All-Reduce

All-Reduce runs Reduce-Scatter followed by All-Gather. The vector is divided into
`N` chunks, and each phase takes `N - 1` steps around the ring:

1. **Reduce-Scatter:** partial results travel around the ring. Rank `r` ends up with
   the fully reduced chunk `(r + 1) mod N`.
2. **All-Gather:** each rank passes the finished chunks along the ring until every
   rank has the whole result.

Each chunk is split into up to 16 pipeline segments. Input and output can be
separate buffers, and the full vector is never copied between them: the first
reduction of each chunk writes straight into `recvbuf`.

### Eager and Rendezvous

| | Eager | Rendezvous |
| --- | --- | --- |
| Operations | Two-sided Send/Recv | RDMA Write with Immediate |
| Reduce-Scatter data | Received into a stage buffer, then reduced into `recvbuf` | Written into a slot of the next rank's ring buffer, then reduced into `recvbuf` |
| Reduce-Scatter pacing | Finishes each step before the next one | Continuous across all steps, limited only by credits and by data dependencies |
| All-Gather data | Received directly into `recvbuf` | Zero-copy RDMA Write into the next rank's `recvbuf` |
| Sync between phases | None | Ring barrier |

### Rendezvous details

- **Credit-based flow control:** the 4 MB ring buffer has 16 slots of 256 KB. A
  sender spends one credit per segment, and the receiver returns credits in batches
  of 4 using zero-byte Send with Immediate messages.
- **Overlapping communication and computation:** the segments of all steps are
  flattened into a single schedule. A segment can be forwarded as soon as the
  reduction it depends on is done, while earlier steps are still in flight.
- **Ring barrier:** an arrival token and then a release token circulate the ring
  over RDMA, so that no rank starts All-Gather before all of them finish
  Reduce-Scatter.
- **Registration cache:** memory regions for user buffers are cached by pointer, so
  reusing a buffer costs only one `ibv_reg_mr()`. If `recvbuf` changes, a rank
  announces the new address and rkey to its previous neighbor with a one-sided RDMA
  Write, without using TCP.
- **Pre-posted receives:** receives for each call are posted up front, which avoids
  RNR stalls.

## API

```c
typedef enum { DT_INT32, DT_FLOAT, DT_DOUBLE } DATATYPE;
typedef enum { OP_SUM, OP_MAX, OP_MIN, OP_PROD } OPERATION;

int connect_process_group(char *servername, void **pg_handle);
int pg_all_reduce_eager(pg_context_t *ctx, const void *sendbuf, void *recvbuf,
                        int count, DATATYPE dt, OPERATION op);
int pg_all_reduce_rendezvous(pg_context_t *ctx, const void *sendbuf, void *recvbuf,
                             int count, DATATYPE dt, OPERATION op);
int pg_close(void *pg_handle);
```

All of these return 0 on success and -1 on failure. The two All-Reduce functions
are `static` in `main.c`. `sendbuf` and `recvbuf` can be the same pointer or
separate, non-overlapping buffers.

`connect_process_group()` ignores `servername`. The ring comes from
`pg_set_topology()`, which `main()` calls with the command-line values first.

## Limits

- At most 4 processes (`MAX_PROCESSES`).
- Buffers are sized for messages of up to 4 MB (`MAX_STAGE_BYTES`). Larger messages
  can fail with an error.
- Uses the first IB device found, on port 1, with an MTU of 1024.
- Addressing is LID based, with no GID configured, so RoCE is not supported.
