"""Internal service-to-service API key verification."""

import os
import secrets

from fastapi import HTTPException, Request


def verify_internal_api_key(request: Request):
    """Validasi header X-API-Key untuk komunikasi backend-to-backend."""
    expected = os.getenv('AI_INTERNAL_SECRET', '').strip()
    if not expected:
        return
    provided = request.headers.get('x-api-key', '')
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=403,
            detail='Akses layanan AI ditolak (X-API-Key tidak valid).',
        )

