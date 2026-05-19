# Remote Development Machine Specifications

## CPU

```bash
grep "model name" /proc/cpuinfo | head -1
    model name : Intel(R) Core(TM) i9-14900K
```

## GPU

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    
    name, memory.total [MiB], driver_version, compute_cap
    NVIDIA GeForce RTX 4090, 24564 MiB, 550.163.01, 8.9
```

## RAM

```bash
sudo dmidecode -t memory | grep -E "Size|Speed|Type:"

    Error Correction Type: None
    Size: 16 GB
    Type: DDR5
    Speed: 4000 MT/s
    Configured Memory Speed: 4000 MT/s
    Non-Volatile Size: None
    Volatile Size: 16 GB
    Cache Size: None
    Logical Size: None
    Size: 16 GB
    Type: DDR5
    Speed: 4000 MT/s
    Configured Memory Speed: 4000 MT/s
    Non-Volatile Size: None
    Volatile Size: 16 GB
    Cache Size: None
    Logical Size: None
    Size: 16 GB
    Type: DDR5
    Speed: 4000 MT/s
    Configured Memory Speed: 4000 MT/s
    Non-Volatile Size: None
    Volatile Size: 16 GB
    Cache Size: None
    Logical Size: None
    Size: 16 GB
    Type: DDR5
    Speed: 4000 MT/s
    Configured Memory Speed: 4000 MT/s
    Non-Volatile Size: None
    Volatile Size: 16 GB
    Cache Size: None
    Logical Size: None
```

## OS

```bash
cat /etc/os-release

    PRETTY_NAME="Debian GNU/Linux 13 (trixie)"
    NAME="Debian GNU/Linux"
    VERSION_ID="13"
    VERSION="13 (trixie)"
    VERSION_CODENAME=trixie
    DEBIAN_VERSION_FULL=13.4
    ID=debian
    HOME_URL="https://www.debian.org/"
    SUPPORT_URL="https://www.debian.org/support"
    BUG_REPORT_URL="https://bugs.debian.org/"
```
