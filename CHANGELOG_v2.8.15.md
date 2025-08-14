# Kangaroo BSGS v2.8.15 Release Notes

## 🚀 Major Features

### Smart Build System with Universal GPU Support
- **Intelligent CUDA Detection**: Automatically detects CUDA availability and builds appropriate version
- **Comprehensive GPU Architecture Support**: Full support for NVIDIA GPUs from Maxwell to Hopper
  - Maxwell: sm_52, sm_53
  - Pascal: sm_60, sm_61
  - Volta: sm_70, sm_72
  - Turing: sm_75
  - Ampere: sm_80, sm_86
  - Ada Lovelace: sm_89
  - Hopper: sm_90
- **CPU-only Fallback**: Seamless compilation and execution on systems without CUDA
- **Cross-platform Compatibility**: Works on both Windows and Linux

### Enhanced Build System
- **Automatic Mode Selection**: 
  - CUDA available → GPU-accelerated version with full architecture support
  - CUDA not available → CPU-only version with identical functionality
- **Conditional Compilation**: Clean separation of GPU and CPU code paths using `#ifdef WITHGPU`
- **Smart CMake Configuration**: Automatically configures build based on system capabilities

## 🔧 Technical Improvements

### Linux Compilation Fixes
- ✅ Fixed all Linux compilation errors
- ✅ Resolved CUDA header dependency issues
- ✅ Added proper conditional compilation for GPU-related code
- ✅ Fixed missing type definitions in CPU-only mode
- ✅ Corrected parameter handling for different build modes

### Code Quality Enhancements
- **Conditional Compilation**: All GPU-related code properly guarded with `#ifdef WITHGPU`
- **Method Overloading**: Separate CPU-only and GPU versions of key methods
- **Error Handling**: Graceful handling of GPU-specific parameters in CPU-only mode
- **Memory Management**: Proper cleanup for both GPU and CPU-only modes

### Build Script Improvements
- **build_linux.sh**: Now auto-detects CUDA and builds appropriate version
- **Intelligent Feedback**: Clear indication of detected build mode
- **Dependency Checking**: Comprehensive validation of build requirements

## 📚 Documentation Updates

### New Documentation Files
- **KANGAROO_COMMAND_REFERENCE_CORRECT.md**: Complete accurate command line reference
- **COMMAND_LINE_ERRORS_EXPLAINED.md**: Detailed explanation of previous documentation errors
- **CHANGELOG_v2.8.15.md**: This release notes document

### Corrected Command Line Documentation
- ✅ Fixed incorrect parameter descriptions
- ✅ Clarified configuration file requirements
- ✅ Updated usage examples for both GPU and CPU-only modes
- ✅ Removed non-existent parameters from documentation

## 🎯 Usage Examples

### Automatic Build (Recommended)
```bash
# Linux - Auto-detects CUDA and builds appropriate version
./build_linux.sh

# Windows - Auto-detects CUDA and builds appropriate version
build_all_windows.bat
```

### Running the Program
```bash
# Basic usage (works in both GPU and CPU-only modes)
./kangaroo -t 8 -d 22 -o result.txt config.txt

# GPU-accelerated (only if CUDA available)
./kangaroo -gpu -gpuId 0,1 -t 8 -d 24 -o result.txt config.txt

# CPU-only mode (always available)
./kangaroo -t 16 -d 20 -o result.txt config.txt
```

### Configuration File Format
```text
<range_start_hex>
<range_end_hex>
<public_key_1_hex>
<public_key_2_hex>
...
```

Example:
```text
100000000000000
1ffffffffffffff
02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea
```

## 🧪 Testing & Validation

### Verified Functionality
- ✅ **Linux WSL Compilation**: Successfully compiles in Ubuntu WSL environment
- ✅ **CPU-only Execution**: Confirmed working at 12+ MK/s on test hardware
- ✅ **Parameter Validation**: All command line parameters work correctly
- ✅ **Configuration Files**: Proper parsing of input configuration files
- ✅ **Enhanced Checkpoints**: Automatic compression system works in both modes

### Performance Metrics
- **CPU-only Mode**: 12.06 MK/s (2 threads, test configuration)
- **Memory Usage**: Efficient memory management in both modes
- **Checkpoint Compression**: 40-70% file size reduction maintained

## 🔄 Backward Compatibility

### Maintained Compatibility
- ✅ All existing command line parameters work unchanged
- ✅ Configuration file format unchanged
- ✅ Checkpoint files fully compatible (legacy and compressed)
- ✅ Network mode (client/server) fully functional
- ✅ Work file management commands unchanged

### Migration Notes
- No migration required - existing setups work without changes
- Enhanced checkpoint compression works automatically
- GPU detection is automatic - no manual configuration needed

## 🐛 Bug Fixes

### Linux Compilation Issues
- Fixed CUDA header inclusion errors
- Resolved missing type definitions in CPU-only mode
- Corrected variable scope issues in conditional compilation
- Fixed CMake configuration for optional CUDA support

### Documentation Corrections
- Removed references to non-existent command line parameters
- Fixed incorrect usage examples
- Clarified configuration file requirements
- Updated parameter descriptions to match actual implementation

## 🚀 System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (2+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **OS**: Windows 10+ or Linux (Ubuntu 18.04+)
- **Compiler**: GCC 7+ (Linux) or MSVC 2019+ (Windows)

### Optional GPU Requirements
- **NVIDIA GPU**: Maxwell architecture or newer (GTX 900 series+)
- **CUDA**: Version 10.0+ (automatically detected)
- **GPU Memory**: 2GB+ recommended for large ranges

## 📈 Performance Improvements

### Build System
- Faster configuration with automatic CUDA detection
- Reduced build time through conditional compilation
- Better error messages and build feedback

### Runtime Performance
- Maintained performance in GPU mode
- Optimized CPU-only execution path
- Enhanced checkpoint compression (40-70% reduction)

## 🔮 Future Roadmap

### Planned Enhancements
- Additional GPU vendor support (AMD, Intel)
- Further performance optimizations
- Enhanced monitoring and statistics
- Improved distributed computing features

---

**Release Date**: January 2025  
**Version**: 2.8.15  
**Compatibility**: Full backward compatibility maintained  
**Recommended Upgrade**: Yes - significant stability and compatibility improvements
