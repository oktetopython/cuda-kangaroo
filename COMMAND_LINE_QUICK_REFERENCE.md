# Kangaroo Command Line Quick Reference

## Essential Commands

### Start New Search
```bash
# Windows
kangaroo.exe -t 8 -d 22 -w search.kcp -i 600 -o result.txt <public_key>

# Linux  
./kangaroo -t 8 -d 22 -w search.kcp -i 600 -o result.txt <public_key>
```

### Resume from Checkpoint
```bash
kangaroo.exe -r search.kcp -o result.txt
```

### Check Checkpoint
```bash
kangaroo.exe -check search.kcp
kangaroo.exe -info search.kcp
```

## Core Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-t <n>` | CPU threads (1-256) | `-t 8` |
| `-d <n>` | Distinguished point bits (16-32) | `-d 22` |
| `-w <file>` | Save checkpoint file | `-w work.kcp` |
| `-r <file>` | Resume from checkpoint | `-r work.kcp` |
| `-i <sec>` | Save interval (60-7200) | `-i 600` |
| `-o <file>` | Output result file | `-o result.txt` |

## GPU Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-gpu` | Enable GPU acceleration | `-gpu` |
| `-gpuId <n>` | GPU device ID | `-gpuId 0` |
| `-gpuId <n,m>` | Multiple GPUs | `-gpuId 0,1` |

## Search Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-r <start:end>` | Search range (hex) | `-r 1000000000:1FFFFFFFF` |
| `-puzzle <n>` | Bitcoin puzzle number | `-puzzle 120` |
| `-maxMem <MB>` | Memory limit | `-maxMem 8192` |
| `-fastMode` | Fast mode | `-fastMode` |

## Checkpoint Management

| Command | Description | Example |
|---------|-------------|---------|
| `-check <file>` | Validate integrity | `-check work.kcp` |
| `-info <file>` | Show file info | `-info work.kcp` |
| `-convert <in> <out>` | Convert format | `-convert old.kcp new.kcp` |
| `-merge <out> <in1> <in2>` | Merge files | `-merge final.kcp work1.kcp work2.kcp` |

## Common Usage Patterns

### Basic Search
```bash
kangaroo.exe -t 4 -d 20 -w basic.kcp -i 300 <public_key>
```

### High Performance
```bash
kangaroo.exe -t 16 -d 24 -w perf.kcp -i 1800 -maxMem 16384 <public_key>
```

### GPU Accelerated
```bash
kangaroo.exe -gpu -gpuId 0 -t 8 -d 24 -w gpu.kcp -i 900 <public_key>
```

### Range Search
```bash
kangaroo.exe -t 8 -d 22 -w range.kcp -r 1000000000:1FFFFFFFF <public_key>
```

### Resume Work
```bash
kangaroo.exe -r existing.kcp -t 8 -o result.txt
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
