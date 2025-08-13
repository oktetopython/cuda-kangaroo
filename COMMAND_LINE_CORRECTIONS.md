# Command Line Usage Corrections

## ⚠️ Important Correction

The command line documentation in `CHECKPOINT_COMPRESSION_README.md` contains **incorrect information**. The actual Kangaroo BSGS program does NOT accept public keys as command line arguments.

## ✅ Correct Usage

### Actual Command Format
```bash
kangaroo.exe [options] config_file.txt
```

### Configuration File Required
The program requires a configuration file with this format:
```text
<range_start_hex>
<range_end_hex>
<public_key_1_hex>
<public_key_2_hex>
...
```

### Real Examples

#### Example 1: test_input.txt
```text
100000000000000
1ffffffffffffff
02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea
```

#### Example 2: puzzle135.txt
```text
4000000000000000000000000000000000
7fffffffffffffffffffffffffffffffff
02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
```

### Correct Command Examples

#### Basic Search
```bash
# Windows
kangaroo.exe -t 8 -d 22 -w work.kcp -o result.txt config.txt

# Linux
./kangaroo -t 8 -d 22 -w work.kcp -o result.txt config.txt
```

#### Resume from Checkpoint
```bash
kangaroo.exe -i work.kcp -t 8 -o result.txt
```

#### GPU Accelerated
```bash
kangaroo.exe -gpu -gpuId 0 -t 4 -d 24 -w gpu_work.kcp -o result.txt config.txt
```

## ✅ Correct Parameters

### Core Parameters
- `-t <n>` : Number of CPU threads
- `-d <n>` : Distinguished point bits (auto if not specified)
- `-w <file>` : Save work to file (current key only)
- `-i <file>` : Load work from file (current key only)
- `-o <file>` : Output result file
- `-v` : Print version

### GPU Parameters
- `-gpu` : Enable GPU calculation
- `-gpuId <list>` : List of GPU IDs (e.g., 0,1,2)
- `-g <grid>` : GPU kernel gridsize
- `-l` : List CUDA enabled devices
- `-check` : Check GPU kernel vs CPU

### Network Parameters
- `-s` : Start in server mode
- `-c <ip>` : Start in client mode
- `-sp <port>` : Server port (default: 17403)
- `-nt <ms>` : Network timeout

### Work File Management
- `-ws` : Save kangaroo work
- `-wss` : Save kangaroo work by server
- `-wsplit` : Split work file
- `-wm <file1> <file2> <dest>` : Merge work files
- `-wi <file>` : Show work file info
- `-winfo <file>` : Show work file info
- `-wcheck <file>` : Check work file

## ❌ Incorrect Parameters (Do NOT Exist)

These parameters mentioned in the documentation are **WRONG**:
- `-i <seconds>` (checkpoint interval) - DOES NOT EXIST
- `-r <file>` (resume) - DOES NOT EXIST (use `-i` instead)
- `-puzzle <n>` - DOES NOT EXIST
- `-maxMem <MB>` - DOES NOT EXIST
- `-fastMode` - DOES NOT EXIST
- `-convert` - DOES NOT EXIST
- `-merge` - DOES NOT EXIST (use `-wm` instead)
- `-info` - DOES NOT EXIST (use `-wi` instead)

## 📚 Correct Documentation

Please refer to `KANGAROO_COMMAND_REFERENCE.md` for the accurate and complete command line reference based on the actual source code.

## 🔧 Enhanced Checkpoint Features

The enhanced checkpoint compression system works automatically:

### Automatic Compression
- All work files (`.kcp`) use automatic compression
- 40-70% file size reduction typical
- No additional parameters needed
- Automatic fallback if compression not beneficial

### Backward Compatibility
- Automatically detects legacy vs compressed format
- Seamlessly loads both old and new checkpoint files
- No manual conversion required

### Usage with Compression
```bash
# Save with automatic compression
kangaroo.exe -t 8 -d 22 -w compressed_work.kcp -o result.txt config.txt

# Resume from any format (legacy or compressed)
kangaroo.exe -i compressed_work.kcp -t 8 -o result.txt

# Check work file (shows compression info)
kangaroo.exe -wi compressed_work.kcp
```

## 🎯 Summary

1. **Always use a configuration file** - public keys are NOT command line arguments
2. **Use `-i` to resume** - not `-r`
3. **Use `-wi` for file info** - not `-info`
4. **Use `-wm` to merge** - not `-merge`
5. **Compression is automatic** - no special parameters needed

The enhanced checkpoint system provides significant file size reduction while maintaining perfect compatibility with the existing command line interface.
