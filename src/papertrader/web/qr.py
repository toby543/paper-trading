"""Renders a TOTP otpauth:// URI as a scannable QR code image.

Uses qrcode's pure-Python PNG backend (the `pypng` package) rather than
Pillow -- Pillow pulls in compiled/binary image libraries that are
overkill just to draw a black-and-white QR code, and can be slow or
fiddly to install on constrained hardware (e.g. a Raspberry Pi). The
image is returned as a data: URI so the enrollment page can embed it
directly with no extra request and no external CDN call -- this is a
login/2FA-setup page, so it shouldn't depend on any third-party network
resource being reachable.
"""
from __future__ import annotations

import base64
import io

import qrcode
from qrcode.image.pure import PyPNGImage


def totp_qr_data_uri(otpauth_uri: str) -> str:
    img = qrcode.make(otpauth_uri, image_factory=PyPNGImage, box_size=6, border=2)
    buf = io.BytesIO()
    img.save(buf)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
