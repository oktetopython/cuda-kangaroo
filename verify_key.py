#!/usr/bin/env python3
"""
Verify that the found private key corresponds to the target public key
"""

import hashlib
import ecdsa
from ecdsa import SigningKey, SECP256k1

def private_key_to_public_key(private_key_hex):
    """Convert private key to compressed public key"""
    # Remove 0x prefix if present
    if private_key_hex.startswith('0x'):
        private_key_hex = private_key_hex[2:]
    
    # Convert to integer
    private_key_int = int(private_key_hex, 16)
    
    # Create signing key
    sk = SigningKey.from_secret_exponent(private_key_int, curve=SECP256k1)
    
    # Get public key point
    vk = sk.get_verifying_key()
    point = vk.pubkey.point
    
    # Convert to compressed format
    x = point.x()
    y = point.y()
    
    # Determine prefix (02 for even y, 03 for odd y)
    prefix = "02" if y % 2 == 0 else "03"
    
    # Format x coordinate as 32-byte hex string
    x_hex = format(x, '064x')
    
    return prefix + x_hex

def main():
    # Test cases
    test_cases = [
        ("A7B", "Found by Kangaroo"),
        ("12345", "Original test case"),
        ("2683", "A7B in decimal"),
    ]
    
    target_pubkey = "038b00fcbfc1a203f44bf123fc7f4c91c10a85c8eae9187f9d22242b4600ce781c"
    
    print("=== Private Key Verification ===")
    print(f"Target public key: {target_pubkey}")
    print()
    
    for private_key, description in test_cases:
        try:
            # Convert to hex if it's decimal
            if private_key.isdigit():
                private_key_hex = hex(int(private_key))[2:]
            else:
                private_key_hex = private_key
                
            computed_pubkey = private_key_to_public_key(private_key_hex)
            
            print(f"{description}:")
            print(f"  Private key: 0x{private_key_hex.upper()}")
            print(f"  Computed pubkey: {computed_pubkey}")
            print(f"  Match: {'✅ YES' if computed_pubkey.lower() == target_pubkey.lower() else '❌ NO'}")
            print()
            
        except Exception as e:
            print(f"{description}: Error - {e}")
            print()

if __name__ == "__main__":
    main()
