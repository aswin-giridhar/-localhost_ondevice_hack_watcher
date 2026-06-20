"""Self-signed TLS so phones/tablets can use their cameras.

Mobile browsers only allow camera access (getUserMedia) on secure origins
(https / localhost). This generates a self-signed certificate that includes the
laptop's LAN IPs as SubjectAltNames, so https://<laptop-ip>:8000/phone works.
Everything is local — the cert never leaves the device.
"""
from __future__ import annotations

import datetime
import ipaddress
import socket
from pathlib import Path


def local_ips() -> list[str]:
    """Best-effort list of this machine's IPv4 addresses."""
    ips: set[str] = {"127.0.0.1"}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))   # no packets sent; just picks the egress IP
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def ensure_cert(cert_path: str, key_path: str) -> tuple[str, str]:
    """Return (cert, key) paths, generating a self-signed pair if missing."""
    cert_p, key_p = Path(cert_path), Path(key_path)
    if cert_p.exists() and key_p.exists():
        return str(cert_p), str(key_p)

    # Imported lazily so the rest of the app runs without `cryptography` installed.
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert_p.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    san: list = [x509.DNSName("localhost")]
    for ip in local_ips():
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            continue

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "watcher.local")])
    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_p.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_p), str(key_p)
