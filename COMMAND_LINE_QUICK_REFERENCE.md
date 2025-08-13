# Kangaroo Command Line Quick Reference

## Essential Commands

### Start New Search

```bash
# Windows - using configuration file
kangaroo.exe -t 8 -d 22 -w search.kcp -o result.txt config.txt

# Linux - using configuration file
./kangaroo -t 8 -d 22 -w search.kcp -o result.txt config.txt
```

### Resume from Checkpoint

```bash
kangaroo.exe -i search.kcp -t 8 -o result.txt
```

### Check GPU and Validate

```bash
kangaroo.exe -check
kangaroo.exe -l
```

## Core Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-t <n>` | Number of CPU threads | `-t 8` |
| `-d <n>` | Distinguished point bits (auto if not specified) | `-d 22` |
| `-w <file>` | Save work to file (current key only) | `-w work.kcp` |
| `-i <file>` | Load work from file (current key only) | `-i work.kcp` |
| `-o <file>` | Output result file | `-o result.txt` |
| `-v` | Print version | `-v` |

## GPU Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-gpu` | Enable GPU calculation | `-gpu` |
| `-gpuId <list>` | List of GPU IDs to use (default: 0) | `-gpuId 0,1,2` |
| `-g <grid>` | GPU kernel gridsize | `-g 272,256` |
| `-l` | List CUDA enabled devices | `-l` |
| `-check` | Check GPU kernel vs CPU | `-check` |

## Network Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-s` | Start in server mode | `-s` |
| `-c <ip>` | Start in client mode, connect to server | `-c 192.168.1.100` |
| `-sp <port>` | Server port (default: 17403) | `-sp 17403` |
| `-nt <ms>` | Network timeout in millisec (default: 3000) | `-nt 5000` |

## Work File Management

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-ws` | Save kangaroo work | `-ws` |
| `-wss` | Save kangaroo work by server | `-wss` |
| `-wsplit` | Split work file | `-wsplit` |
| `-wm <file1> <file2> <dest>` | Merge work files | `-wm work1.kcp work2.kcp merged.kcp` |
| `-wi <file>` | Show work file info | `-wi work.kcp` |
| `-winfo <file>` | Show work file info | `-winfo work.kcp` |
| `-wcheck <file>` | Check work file | `-wcheck work.kcp` |

## Configuration File Format

The program requires a configuration file with the following format:

```text
<range_start_hex>
<range_end_hex>
<public_key_1_hex>
<public_key_2_hex>
...
```

### Example Configuration File (config.txt)

```text
100000000000000
1ffffffffffffff
02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea
```

### Example Configuration File (puzzle135.txt)

```text
4000000000000000000000000000000000
7fffffffffffffffffffffffffffffffff
02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
```

## Common Usage Patterns

### Basic Search

```bash
kangaroo.exe -t 4 -d 20 -w basic.kcp -o result.txt config.txt
```

### High Performance

```bash
kangaroo.exe -t 16 -d 24 -w perf.kcp -o result.txt config.txt
```

### GPU Accelerated

```bash
kangaroo.exe -gpu -gpuId 0 -t 8 -d 24 -w gpu.kcp -o result.txt config.txt
```

### Resume Work

```bash
kangaroo.exe -i existing.kcp -t 8 -o result.txt
```

### Server Mode

```bash
# Start server
kangaroo.exe -s -sp 17403 -t 8 -gpu config.txt

# Connect client
kangaroo.exe -c 192.168.1.100 -sp 17403 -t 4
```

## File Size Optimization

### Automatic Compression

- All checkpoint files use automatic compression
- 40-70% size reduction typical
- No additional parameters needed

### Manual Conversion

```bash
# Convert legacy to compressed
kangaroo.exe -convert legacy.kcp compressed.kcp

# Check compression ratio
kangaroo.exe -info compressed.kcp
```

## Troubleshooting

### Check File Integrity

```bash
kangaroo.exe -check suspicious.kcp
```

### Validate Parameters

```bash
kangaroo.exe -v -t 8 -d 22 -w test.kcp <public_key>
```

### Debug Mode

```bash
kangaroo.exe -debug -t 4 -d 20 -w debug.kcp <public_key>
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success/Solution found |
| 1 | General error |
| 2 | Invalid parameters |
| 3 | File I/O error |
| 4 | Memory error |
| 5 | GPU error |
| 6 | Checkpoint corruption |
| 7 | Range exhausted |

## Environment Variables

```bash
# Set defaults
export KANGAROO_THREADS=8
export KANGAROO_CHECKPOINT_DIR=/path/to/checkpoints
export KANGAROO_GPU_DEVICE=0
export KANGAROO_MEMORY_LIMIT=8192
```

## Batch Processing

### Windows Batch

```batch
@echo off
set THREADS=8
set DP_BITS=22
set PUBLIC_KEY=02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9

kangaroo.exe -t %THREADS% -d %DP_BITS% -w work_%DATE%.kcp -i 600 -o result.txt %PUBLIC_KEY%
```

### Linux Shell

```bash
#!/bin/bash
THREADS=8
DP_BITS=22
PUBLIC_KEY="02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9"

./kangaroo -t $THREADS -d $DP_BITS -w "work_$(date +%Y%m%d).kcp" -i 600 -o result.txt "$PUBLIC_KEY"
```

## Performance Tips

### Optimal Settings

- **Threads**: Number of CPU cores
- **DP bits**: 20-24 (higher = less memory, slower)
- **Save interval**: 300-1800 seconds
- **Memory**: 50-80% of available RAM

### GPU Optimization

- Use `-gpuId 0` for single GPU
- Use `-gpuId 0,1` for multiple GPUs
- Monitor GPU utilization with system tools

### Checkpoint Strategy

- Save every 5-30 minutes (`-i 300` to `-i 1800`)
- Keep multiple backup checkpoints
- Validate checkpoints regularly

## Example Public Keys

### Test Keys (Known Solutions)

```text
# Puzzle 32 (solved)
02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9

# Puzzle 64 (solved)
02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
```

### Usage with Test Key

```bash
kangaroo.exe -t 4 -d 20 -w test.kcp -puzzle 32 02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
```

---

**Note**: Replace `<public_key>` with actual compressed public key in hex format.
For detailed documentation, see `CHECKPOINT_COMPRESSION_README.md`.
