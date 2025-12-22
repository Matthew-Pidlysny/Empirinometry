# Diffusion Navigator Installation Guide
## Enhanced Edition Version 3.0

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Detailed Installation Instructions](#detailed-installation-instructions)
4. [Configuration](#configuration)
5. [Troubleshooting Installation](#troubleshooting-installation)
6. [Post-Installation Setup](#post-installation-setup)
7. [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| **Operating System** | Windows 10, macOS 10.14, Ubuntu 18.04+ |
| **Processor** | Intel Core i3 or AMD equivalent (64-bit) |
| **Memory (RAM)** | 4 GB DDR4 |
| **Storage** | 500 MB available disk space |
| **Graphics** | OpenGL 3.3 compatible graphics card |
| **Python** | Version 3.11 or higher |
| **Network** | Internet connection for initial setup |

### Recommended Requirements

| Component | Recommended Specification |
|-----------|---------------------------|
| **Operating System** | Windows 11, macOS 12+, Ubuntu 20.04+ |
| **Processor** | Intel Core i5/i7 or AMD Ryzen 5/7 (64-bit) |
| **Memory (RAM)** | 8 GB DDR4 or higher |
| **Storage** | 2 GB available disk space (for cache and data) |
| **Graphics** | Dedicated graphics card with 2GB+ VRAM |
| **Python** | Version 3.11+ with virtual environment |
| **Network** | Broadband internet connection |

### Platform-Specific Requirements

#### Windows
- **Visual C++ Redistributable**: 2019 or later
- **.NET Framework**: Version 4.8 or later
- **Windows Update**: Latest updates installed

#### macOS
- **Xcode Command Line Tools**: Required for some dependencies
- **macOS Updates**: Latest security updates
- **Gatekeeper**: Allow applications from identified developers

#### Linux
- **Build Tools**: `build-essential` (Ubuntu/Debian) or equivalent
- **OpenGL Development**: `libgl1-mesa-dev` and related packages
- **Python Development**: `python3-dev` package

---

## Installation Methods

### Method 1: Pre-compiled Binary Package (Recommended)

**Best for**: Most users, educational institutions, production environments

**Advantages**:
- No installation of dependencies required
- Ready to run immediately
- All components tested together
- Includes GUI and all features

**Download Options**:
- **Windows Installer**: `DiffusionNavigator-v3.0-Windows.exe` (85 MB)
- **macOS DMG**: `DiffusionNavigator-v3.0-macOS.dmg` (92 MB)
- **Linux AppImage**: `DiffusionNavigator-v3.0-Linux.AppImage` (88 MB)

### Method 2: Python Package Installation

**Best for**: Developers, researchers, custom installations

**Advantages**:
- Flexible installation options
- Can be integrated into existing Python environments
- Source code access for customization
- Easier updates and version management

**Installation Commands**:
```bash
# Using pip (recommended)
pip install diffusion-navigator

# Using conda
conda install -c conda-forge diffusion-navigator

# Development installation
pip install diffusion-navigator[dev]
```

### Method 3: Source Code Installation

**Best for**: Advanced users, contributors, custom modifications

**Advantages**:
- Full control over installation
- Access to latest development features
- Ability to modify source code
- Suitable for debugging and development

**Repository**: `https://github.com/diffusion-navigator/diffusion-navigator`

---

## Detailed Installation Instructions

### Option 1: Windows Installation

#### Method A: Installer Package

1. **Download the installer**:
   - Visit diffusion-navigator.org/downloads
   - Download `DiffusionNavigator-v3.0-Windows.exe`
   - Verify download integrity (optional but recommended)

2. **Run the installer**:
   - Right-click the installer and select "Run as administrator"
   - Follow the installation wizard
   - Choose installation directory (default: `C:\Program Files\Diffusion Navigator`)
   - Select components to install (Full installation recommended)

3. **Desktop shortcut creation**:
   - Check "Create desktop shortcut" for easy access
   - Check "Add to PATH" if you want command-line access

4. **Complete installation**:
   - Click "Install" and wait for completion
   - Launch the application from desktop shortcut or Start menu

#### Method B: Python Package

1. **Install Python** (if not already installed):
   - Download Python 3.11+ from python.org
   - During installation, check "Add Python to PATH"
   - Verify installation: Open Command Prompt and run `python --version`

2. **Install Diffusion Navigator**:
   ```cmd
   pip install diffusion-navigator
   ```

3. **Launch the application**:
   ```cmd
   diffusion-navigator
   ```
   or
   ```cmd
   python -m diffusion_navigator
   ```

### Option 2: macOS Installation

#### Method A: DMG Package

1. **Download the DMG**:
   - Visit diffusion-navigator.org/downloads
   - Download `DiffusionNavigator-v3.0-macOS.dmg`
   - Verify download if desired

2. **Install the application**:
   - Double-click the DMG file to mount it
   - Drag Diffusion Navigator to Applications folder
   - Eject the DMG after installation

3. **First launch**:
   - Open Applications folder and double-click Diffusion Navigator
   - If you see a security warning, go to System Preferences → Security & Privacy
   - Click "Open Anyway" to allow the application to run

4. **Add to Dock** (optional):
   - Right-click the dock icon and select "Keep in Dock"

#### Method B: Python Package

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**:
   ```bash
   brew install python@3.11
   ```

3. **Install Diffusion Navigator**:
   ```bash
   pip3 install diffusion-navigator
   ```

4. **Launch the application**:
   ```bash
   diffusion-navigator
   ```

### Option 3: Linux Installation

#### Method A: AppImage (Universal)

1. **Download the AppImage**:
   - Visit diffusion-navigator.org/downloads
   - Download `DiffusionNavigator-v3.0-Linux.AppImage`

2. **Make the file executable**:
   ```bash
   chmod +x DiffusionNavigator-v3.0-Linux.AppImage
   ```

3. **Run the application**:
   ```bash
   ./DiffusionNavigator-v3.0-Linux.AppImage
   ```

4. **Create desktop entry** (optional):
   ```bash
   ./DiffusionNavigator-v3.0-Linux.AppImage --install-desktop
   ```

#### Method B: Package Manager

**Ubuntu/Debian**:
```bash
# Add repository
sudo add-apt-repository ppa:diffusion-navigator/ppa
sudo apt update

# Install
sudo apt install diffusion-navigator
```

**Fedora/CentOS**:
```bash
# Enable EPEL repository (for CentOS/RHEL)
sudo dnf install epel-release

# Install
sudo dnf install diffusion-navigator
```

#### Method C: Python Package

1. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv python3-dev
   sudo apt install build-essential libgl1-mesa-dev libglu1-mesa-dev
   
   # Fedora/CentOS
   sudo dnf install python3 python3-pip python3-devel
   sudo dnf groupinstall "Development Tools"
   sudo dnf install mesa-libGL-devel mesa-libGLU-devel
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv diffusion_navigator_env
   source diffusion_navigator_env/bin/activate
   ```

3. **Install Diffusion Navigator**:
   ```bash
   pip install diffusion-navigator
   ```

4. **Launch the application**:
   ```bash
   diffusion-navigator
   ```

### Option 4: Source Code Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/diffusion-navigator/diffusion-navigator.git
   cd diffusion-navigator
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

5. **Run the application**:
   ```bash
   python diffusion_navigator/main.py
   ```

---

## Configuration

### Environment Variables

Optional environment variables for customization:

```bash
# Cache directory
export DIFFUSION_NAVIGATOR_CACHE_DIR="/path/to/cache"

# Data directory
export DIFFUSION_NAVIGATOR_DATA_DIR="/path/to/data"

# Log level (DEBUG, INFO, WARNING, ERROR)
export DIFFUSION_NAVIGATOR_LOG_LEVEL="INFO"

# GPU acceleration (enabled/disabled)
export DIFFUSION_NAVIGATOR_GPU_ACCELERATION="enabled"
```

### Configuration Files

Configuration is stored in platform-specific locations:

#### Windows
```
C:\Users\%USERNAME%\AppData\Roaming\DiffusionNavigator\config.json
```

#### macOS
```
~/Library/Application Support/DiffusionNavigator/config.json
```

#### Linux
```
~/.config/diffusion-navigator/config.json
```

### Sample Configuration

```json
{
  "application": {
    "auto_save": true,
    "show_tips": true,
    "default_units": "si",
    "theme": "default"
  },
  "visualization": {
    "high_quality": true,
    "animations_enabled": true,
    "color_scheme": "scientific",
    "max_points": 10000
  },
  "performance": {
    "cache_enabled": true,
    "cache_size_mb": 500,
    "parallel_processing": true,
    "max_workers": 4
  },
  "advanced": {
    "debug_mode": false,
    "experimental_features": false,
    "custom_materials": []
  }
}
```

---

## Troubleshooting Installation

### Common Installation Issues

#### Issue 1: Python Not Found

**Symptoms**:
- `python: command not found` error
- Python version too old

**Solutions**:

**Windows**:
1. Download and install Python 3.11+ from python.org
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt and verify with `python --version`

**macOS**:
```bash
# Using Homebrew
brew install python@3.11

# Using pyenv
brew install pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora/CentOS
sudo dnf install python3 python3-pip python3-venv
```

#### Issue 2: Permission Denied

**Symptoms**:
- Permission denied during installation
- Cannot write to installation directory

**Solutions**:

**Windows**:
1. Run Command Prompt as administrator
2. Use `--user` flag: `pip install --user diffusion-navigator`
3. Install to user directory instead of system-wide

**macOS/Linux**:
```bash
# Use pip with --user flag
pip install --user diffusion-navigator

# Or use sudo (not recommended)
sudo pip install diffusion-navigator

# Best: Use virtual environment
python -m venv ~/.local/share/diffusion-navigator
source ~/.local/share/diffusion-navigator/bin/activate
pip install diffusion-navigator
```

#### Issue 3: Missing System Dependencies

**Symptoms**:
- Compilation errors during installation
- Missing header files
- OpenGL-related errors

**Solutions**:

**Windows**:
1. Install Visual Studio Build Tools
2. Install Visual C++ Redistributable 2019+
3. Update graphics drivers

**macOS**:
```bash
# Install Xcode command line tools
xcode-select --install

# If issues persist, install full Xcode from App Store
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
sudo apt install libqt6-dev libqt6opengl6-dev

# Fedora/CentOS
sudo dnf groupinstall "Development Tools"
sudo dnf install mesa-libGL-devel mesa-libGLU-devel
sudo dnf install qt6-qtbase-devel qt6-qttools-devel
```

#### Issue 4: Graphics/OpenGL Issues

**Symptoms**:
- Application starts but shows blank windows
- OpenGL version errors
- Visualization not working

**Solutions**:

1. **Update graphics drivers**:
   - Windows: Update through Device Manager or manufacturer website
   - macOS: Use Software Update
   - Linux: Update through package manager or manufacturer PPA

2. **Check OpenGL support**:
   ```bash
   # Linux
   glxinfo | grep "OpenGL version"
   
   # Windows/Mac: Use OpenGL Extensions Viewer
   ```

3. **Software rendering fallback** (Linux):
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   diffusion-navigator
   ```

#### Issue 5: Application Won't Start

**Symptoms**:
- Application crashes on startup
- Error messages but no GUI
- Process starts but exits immediately

**Solutions**:

1. **Run from command line to see errors**:
   ```bash
   diffusion-navigator --verbose
   ```

2. **Check log files**:
   - Windows: `%APPDATA%\DiffusionNavigator\logs\`
   - macOS: `~/Library/Logs/DiffusionNavigator/`
   - Linux: `~/.local/share/diffusion-navigator/logs/`

3. **Try safe mode**:
   ```bash
   diffusion-navigator --safe-mode
   ```

4. **Reset configuration**:
   ```bash
   diffusion-navigator --reset-config
   ```

### Platform-Specific Troubleshooting

#### Windows Specific

**Issue**: "DLL not found" errors
```cmd
# Solution: Install Visual C++ Redistributable
# Download from: https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
```

**Issue**: Windows Defender blocking installation
```cmd
# Solution: Add exception for Diffusion Navigator
# or temporarily disable real-time protection
```

#### macOS Specific

**Issue**: "App can't be opened because it's from an unidentified developer"
```bash
# Solution: Allow app in System Preferences
# or use command: xattr -d com.apple.quarantine /Applications/DiffusionNavigator.app
```

**Issue**: Missing Xcode tools
```bash
xcode-select --install
```

#### Linux Specific

**Issue**: Missing shared libraries
```bash
# Find missing libraries
ldd /usr/bin/diffusion-navigator

# Install missing packages
sudo apt install $(ldd /usr/bin/diffusion-navigator | grep "not found" | awk '{print $3}' | xargs)
```

**Issue**: Wayland display issues
```bash
# Try X11 backend
export QT_QPA_PLATFORM=xcb
diffusion-navigator
```

---

## Post-Installation Setup

### First Launch Configuration

1. **Launch the application**:
   - Double-click desktop shortcut or run from command line
   - Wait for initial setup to complete

2. **Choose preferences**:
   - Select default units (SI or Imperial)
   - Choose theme and color scheme
   - Set cache location and size

3. **Verify installation**:
   - Check that material database loads
   - Test basic calculation functionality
   - Verify visualizations work correctly

### Material Database Setup

1. **Load default materials**:
   - Database should load automatically on first launch
   - Verify 50+ materials are available

2. **Check material data integrity**:
   - Go to Tools → Database → Verify
   - Fix any reported issues

3. **Optional: Add custom materials**:
   - Use Tools → Database → Add Material
   - Follow the material data format guidelines

### Performance Optimization

1. **Cache configuration**:
   - Set appropriate cache size (recommended: 500 MB)
   - Choose cache location on fast storage (SSD preferred)

2. **Parallel processing**:
   - Enable multi-core processing for batch calculations
   - Set worker threads based on CPU cores

3. **Graphics optimization**:
   - Choose appropriate quality settings based on hardware
   - Enable GPU acceleration if available

### Testing Installation

1. **Basic functionality test**:
   - Select "Aluminum" from material dropdown
   - Set temperature to 800 K
   - Set time to 3600 s
   - Click "Calculate Diffusion"
   - Verify results appear in Results tab

2. **Visualization test**:
   - Go to Visualizations tab
   - Select "Diffusion Sphere" visualization type
   - Verify 3D visualization appears correctly

3. **Export test**:
   - Click "Export JSON" button
   - Save test file
   - Verify file is created and contains data

---

## Uninstallation

### Windows

#### Method A: Using Add/Remove Programs

1. Open **Control Panel** → **Programs and Features**
2. Find "Diffusion Navigator" in the list
3. Right-click and select "Uninstall"
4. Follow the uninstallation wizard
5. Choose whether to remove user data (recommended for complete removal)

#### Method B: Manual Uninstallation

```cmd
# Uninstall Python package
pip uninstall diffusion-navigator

# Remove user data (optional)
rmdir /s "%APPDATA%\DiffusionNavigator"
rmdir /s "%LOCALAPPDATA%\DiffusionNavigator"
```

### macOS

#### Method A: Using Applications Folder

1. Open **Applications** folder
2. Drag "Diffusion Navigator" to Trash
3. Empty Trash

#### Method B: Command Line Uninstallation

```bash
# Uninstall Python package
pip3 uninstall diffusion-navigator

# Remove application support files
rm -rf ~/Library/Application\ Support/DiffusionNavigator
rm -rf ~/Library/Logs/DiffusionNavigator
rm -rf ~/Library/Preferences/org.diffusion-navigator.plist
```

### Linux

#### Method A: Package Manager

```bash
# Ubuntu/Debian
sudo apt remove diffusion-navigator
sudo apt autoremove

# Fedora/CentOS
sudo dnf remove diffusion-navigator
```

#### Method B: Python Package

```bash
# Uninstall
pip uninstall diffusion-navigator

# Remove user data (optional)
rm -rf ~/.config/diffusion-navigator
rm -rf ~/.local/share/diffusion-navigator
```

### Clean Uninstallation (All Platforms)

To completely remove all traces of Diffusion Navigator:

1. **Uninstall the main application** (using methods above)
2. **Remove user data directories**:
   - Configuration files
   - Cache directories
   - Log files
   - Custom materials
3. **Remove environment variables** (if set)
4. **Clean up desktop shortcuts and menu entries**

### Data Backup Before Uninstallation

If you want to preserve your data:

```bash
# Create backup directory
mkdir diffusion_navigator_backup

# Backup configuration
cp -r ~/.config/diffusion-navigator/config.json diffusion_navigator_backup/

# Backup custom materials
cp -r ~/.config/diffusion-navigator/custom_materials diffusion_navigator_backup/

# Backup saved calculations
cp -r ~/.local/share/diffusion-navigator/saved_calculations diffusion_navigator_backup/
```

---

## Verification and Validation

### Installation Verification Script

Create a verification script to test your installation:

```python
#!/usr/bin/env python3
"""
Diffusion Navigator Installation Verification Script
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    try:
        from diffusion_navigator import (
            CaelumDiffusionModel,
            EnhancedMaterialDatabase,
            AdvancedDiffusionVisualizer
        )
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    try:
        from diffusion_navigator import CaelumDiffusionModel
        
        model = CaelumDiffusionModel()
        result = model.calculate_diffusion(
            material={'name': 'Test', 'D0': 1e-4, 'Q': 100},
            temperature=800,
            time=3600,
            distance=1e-6
        )
        
        assert 'diffusion_coefficient' in result
        print("✓ Basic calculation test passed")
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_database():
    """Test material database."""
    try:
        from diffusion_navigator import EnhancedMaterialDatabase
        
        db = EnhancedMaterialDatabase()
        materials = db.get_all_materials()
        
        assert len(materials) > 0
        print(f"✓ Database test passed ({len(materials)} materials)")
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Diffusion Navigator Installation Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_database
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print(" Installation verified successfully!")
        return 0
    else:
        print("INCOMPLETE Installation verification failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run the verification script:
```bash
python verify_installation.py
```

---

## Getting Help

### Official Support Channels

- **Documentation**: diffusion-navigator.org/docs
- **Installation Guide**: diffusion-navigator.org/installation
- **FAQ**: diffusion-navigator.org/faq
- **Community Forum**: diffusion-navigator.org/forum
- **Bug Reports**: diffusion-navigator.org/issues
- **Email Support**: support@diffusion-navigator.org

### Community Resources

- **GitHub Repository**: github.com/diffusion-navigator/diffusion-navigator
- **Discord Server**: discord.gg/diffusion-navigator
- **Stack Overflow**: Questions tagged `diffusion-navigator`

### Installation Support

When reporting installation issues, please include:

1. **Operating System**: Windows 10/11, macOS version, Linux distribution
2. **Python Version**: `python --version`
3. **Installation Method**: Installer, pip, source code
4. **Error Messages**: Complete error output
5. **System Specifications**: CPU, RAM, GPU information

### System Information Gathering

For better support, gather system information:

**Windows**:
```cmd
systeminfo > system_info.txt
python --version > python_info.txt
pip list > packages.txt
```

**macOS/Linux**:
```bash
uname -a > system_info.txt
python --version > python_info.txt
pip list > packages.txt
```

---

*This installation guide covers Diffusion Navigator Enhanced Edition v3.0. For the most current installation instructions, visit the official website at diffusion-navigator.org/installation.*