import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "mindtrace-ai"
JWT_EXPIRE_MINUTES = 30
MINDTRACE_ENV = os.getenv("MINDTRACE_ENV", "development").strip().lower()
JWT_SECRET = os.getenv("MINDTRACE_JWT_SECRET", "").strip()
if not JWT_SECRET:
    if MINDTRACE_ENV == "production":
        raise RuntimeError("MINDTRACE_JWT_SECRET is required when MINDTRACE_ENV=production.")
    JWT_SECRET = "development-only-change-this-secret"

password_hash = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, encoded_password: str) -> bool:
    return password_hash.verify(password, encoded_password)


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iss": JWT_ISSUER,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _authentication_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _authentication_error()

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
        )
        if payload.get("type") != "access":
            raise _authentication_error()
        user_id = int(payload["sub"])
    except HTTPException:
        raise
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        raise _authentication_error() from None

    from app import main

    main.init_db()
    with sqlite3.connect(str(main.DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, email, display_name, created_at, updated_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        raise _authentication_error()
    return dict(row)