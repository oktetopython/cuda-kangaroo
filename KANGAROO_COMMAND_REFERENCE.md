# Kangaroo BSGS Command Line Reference

## Overview

Kangaroo is a BSGS (Baby-step Giant-step) implementation for solving discrete logarithm problems. It requires a configuration file specifying the search range and public keys.

## Basic Usage

```bash
# Windows
kangaroo.exe [options] config_file.txt

# Linux
./kangaroo [options] config_file.txt
```

## Configuration File Format

The configuration file must contain:
```text
<range_start_hex>
<range_end_hex>
<public_key_1_hex>
<public_key_2_hex>
...
```

### Example Configuration Files

#### test_input.txt (Small range for testing)
```text
100000000000000
1ffffffffffffff
02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea
```

#### puzzle135.txt (Bitcoin Puzzle 135)
```text
4000000000000000000000000000000000
7fffffffffffffffffffffffffffffffff
02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
```

## Command Line Parameters

### Core Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-t <n>` | Number of CPU threads | `-t 8` |
| `-d <n>` | Distinguished point bits (auto if not specified) | `-d 22` |
| `-w <file>` | Save work to file (current key only) | `-w work.kcp` |
| `-i <file>` | Load work from file (current key only) | `-i work.kcp` |
| `-o <file>` | Output result file | `-o result.txt` |
| `-v` | Print version | `-v` |

### GPU Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-gpu` | Enable GPU calculation | `-gpu` |
| `-gpuId <list>` | List of GPU IDs to use (default: 0) | `-gpuId 0,1,2` |
| `-g <grid>` | GPU kernel gridsize | `-g 272,256` |
| `-l` | List CUDA enabled devices | `-l` |
| `-check` | Check GPU kernel vs CPU | `-check` |

### Network Parameters (Client/Server Mode)
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-s` | Start in server mode | `-s` |
| `-c <ip>` | Start in client mode, connect to server | `-c 192.168.1.100` |
| `-sp <port>` | Server port (default: 17403) | `-sp 17403` |
| `-nt <ms>` | Network timeout in millisec (default: 3000) | `-nt 5000` |

### Work File Management
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-ws` | Save kangaroo work | `-ws` |
| `-wss` | Save kangaroo work by server | `-wss` |
| `-wsplit` | Split work file | `-wsplit` |
| `-wm <file1> <file2> <dest>` | Merge work files | `-wm work1.kcp work2.kcp merged.kcp` |
| `-wi <file>` | Show work file info | `-wi work.kcp` |
| `-winfo <file>` | Show work file info | `-winfo work.kcp` |
| `-wcheck <file>` | Check work file | `-wcheck work.kcp` |

## Common Usage Examples

### Basic Search
```bash
# Start new search
kangaroo.exe -t 8 -d 22 -w search.kcp -o result.txt config.txt

# Resume from checkpoint
kangaroo.exe -i search.kcp -t 8 -o result.txt
```

### GPU Accelerated Search
```bash
# Single GPU
kangaroo.exe -gpu -gpuId 0 -t 4 -d 24 -w gpu_search.kcp -o result.txt config.txt

# Multiple GPUs
kangaroo.exe -gpu -gpuId 0,1,2 -t 8 -d 24 -w multi_gpu.kcp -o result.txt config.txt
```

### Server/Client Mode
```bash
# Start server
kangaroo.exe -s -sp 17403 -t 8 -gpu -w server.kcp config.txt

# Connect client
kangaroo.exe -c 192.168.1.100 -sp 17403 -t 4 -gpu
```

### Work File Management
```bash
# Check work file info
kangaroo.exe -wi work.kcp

# Validate work file
kangaroo.exe -wcheck work.kcp

# Merge multiple work files
kangaroo.exe -wm work1.kcp work2.kcp work3.kcp merged.kcp

# Split work file
kangaroo.exe -wsplit large_work.kcp
```

## Real Examples

### Bitcoin Puzzle 135
```bash
# Create puzzle135.txt file first, then:
kangaroo.exe -t 16 -d 24 -gpu -gpuId 0,1 -w puzzle135.kcp -o result.txt puzzle135.txt
```

### Test Search (Known Solution)
```bash
# Create test_input.txt file first, then:
kangaroo.exe -t 4 -d 20 -w test.kcp -o result.txt test_input.txt
```

### Resume Interrupted Work
```bash
# Resume from existing checkpoint
kangaroo.exe -i puzzle135.kcp -t 16 -gpu -o result.txt
```

## System Information

### Check GPU Support
```bash
# List available CUDA devices
kangaroo.exe -l

# Check GPU vs CPU calculation
kangaroo.exe -check

# Show version
kangaroo.exe -v
```

## Performance Tips

### Optimal Settings
- **Threads**: Use number of CPU cores (e.g., `-t 8` for 8-core CPU)
- **DP bits**: 20-24 for most cases (higher = less memory, slower)
- **GPU**: Always use if available (`-gpu -gpuId 0`)
- **Work files**: Save frequently to prevent loss (`-w work.kcp`)

### Memory Considerations
- Higher DP bits = less memory usage but slower
- Lower DP bits = more memory usage but faster
- Monitor system memory usage during operation

### Network Mode Benefits
- Distribute work across multiple machines
- Server manages work distribution
- Clients can join/leave dynamically

## Output Format

### Result File (result.txt)
When a solution is found, the result file contains:
```text
Key# 0
Pub:  02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea
Priv: 1234567890abcdef
```

### Work File Info (-wi output)
```text
Version   : 2.0
DP bits   : 22
Start     : 100000000000000
Stop      : 1ffffffffffffff
Keys      : 1
[00] Pub : 02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea
Kangaroos : 1048576
```

## Troubleshooting

### Common Issues
1. **No CUDA devices found**: Install NVIDIA drivers and CUDA toolkit
2. **Out of memory**: Increase DP bits (`-d 24` instead of `-d 20`)
3. **Work file corruption**: Use `-wcheck` to validate
4. **Network timeout**: Increase timeout with `-nt 5000`

### Debug Commands
```bash
# Check system compatibility
kangaroo.exe -l
kangaroo.exe -check

# Validate work file
kangaroo.exe -wcheck work.kcp

# Show work file details
kangaroo.exe -wi work.kcp
```

---

**Note**: Always create a proper configuration file before running. The program will not work without a valid config file containing the search range and public keys.
