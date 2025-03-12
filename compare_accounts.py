#!/usr/bin/env python3
"""
Script to compare environment and permissions between different user accounts
"""

import os
import sys
import pwd
import grp
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_system_info():
    """Get detailed system and user information"""
    info = {}
    
    # User info
    info['user'] = {
        'username': os.getlogin(),
        'uid': os.getuid(),
        'gid': os.getgid(),
        'groups': [g.gr_name for g in grp.getgrall() if os.getlogin() in g.gr_mem],
        'home': os.path.expanduser('~'),
        'shell': os.environ.get('SHELL', '')
    }
    
    # Environment variables
    info['env'] = {
        'HF_HOME': os.environ.get('HF_HOME', '(not set)'),
        'HF_DATASETS_CACHE': os.environ.get('HF_DATASETS_CACHE', '(not set)'),
        'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE', '(not set)'),
        'TMPDIR': os.environ.get('TMPDIR', '(not set)'),
        'PATH': os.environ.get('PATH', '(not set)'),
        'PYTHONPATH': os.environ.get('PYTHONPATH', '(not set)'),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', '(not set)')
    }
    
    # Python info
    info['python'] = {
        'executable': sys.executable,
        'version': sys.version,
        'path': sys.path
    }
    
    return info

def check_file_permissions(path):
    """Check detailed permissions and ownership of a file or directory"""
    try:
        stat = os.stat(path)
        info = {
            'path': path,
            'exists': True,
            'type': 'directory' if os.path.isdir(path) else 'file',
            'mode': oct(stat.st_mode)[-3:],  # Permission bits in octal
            'owner': pwd.getpwuid(stat.st_uid).pw_name,
            'group': grp.getgrgid(stat.st_gid).gr_name,
            'uid': stat.st_uid,
            'gid': stat.st_gid
        }
        
        # Check if current user can read/write
        info['user_can_read'] = os.access(path, os.R_OK)
        info['user_can_write'] = os.access(path, os.W_OK)
        info['user_can_execute'] = os.access(path, os.X_OK)
        
        return info
    except Exception as e:
        return {
            'path': path,
            'exists': False,
            'error': str(e)
        }

def check_filesystem_type(path):
    """Get filesystem type for a given path"""
    try:
        result = subprocess.run(['df', '-T', path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting filesystem type: {e}"

def check_lustre_settings(path):
    """Check Lustre-specific settings if the path is on a Lustre filesystem"""
    try:
        # Check if lfs command exists
        if subprocess.run(['which', 'lfs'], capture_output=True).returncode == 0:
            result = subprocess.run(['lfs', 'getstripe', path], capture_output=True, text=True)
            return result.stdout
        return "lfs command not found"
    except Exception as e:
        return f"Error checking Lustre settings: {e}"

def check_loaded_modules():
    """Check which modules are loaded (if using module system)"""
    try:
        result = subprocess.run(['module', 'list'], capture_output=True, text=True)
        return result.stdout
    except Exception:
        return "Module command not available"

def main():
    # Check data directory permissions
    data_dir = "/lustre/fast/fast/wliu/zqiu/GaLore/c4/en"
    logger.info("\n=== Data Directory Permissions ===")
    perms = check_file_permissions(data_dir)
    for k, v in perms.items():
        logger.info(f"{k}: {v}")
    
    # Check filesystem type
    logger.info("\n=== Filesystem Information ===")
    fs_info = check_filesystem_type(data_dir)
    logger.info(f"Filesystem details:\n{fs_info}")
    
    # Check Lustre settings
    logger.info("\n=== Lustre Settings ===")
    lustre_info = check_lustre_settings(data_dir)
    logger.info(f"Lustre stripe info:\n{lustre_info}")
    
    # Check loaded modules
    logger.info("\n=== Loaded Modules ===")
    modules = check_loaded_modules()
    logger.info(modules)
    
    # Try to create a test file
    test_file = os.path.join(data_dir, f"test_file_{os.getlogin()}.txt")
    logger.info("\n=== File Creation Test ===")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        logger.info(f"Successfully created test file: {test_file}")
        os.remove(test_file)
        logger.info("Successfully removed test file")
    except Exception as e:
        logger.info(f"Failed to create/remove test file: {e}")

if __name__ == "__main__":
    main()