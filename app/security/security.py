# 安全工具
import os
from datetime import timezone, datetime, timedelta
from typing import Any

from jwt import encode, decode
from passlib.context import CryptContext

PWD_CONTEXT = CryptContext(
    schemes=["bcrypt_sha256"],
    deprecated="auto"
)
# JWT json web token 是一个三段论（头部签名载荷分别是什么）
JWT_SECRET = os.getenv("JWT_SECRET","dev-only-changeme")
JWT_ALG = os.getenv("JWT_ALG","HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 120))

def hash_password(password) -> str:
    return PWD_CONTEXT.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return PWD_CONTEXT.verify(plain_password, hashed_password)

def create_access_token(payload: dict[str, Any], expires_minutes: int | None = None) -> str:
    minutes = expires_minutes or JWT_EXPIRE_MINUTES
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    to_encode = {**payload, "exp": exp}
    return encode(to_encode, JWT_SECRET, JWT_ALG)

def decode_token(token: str) -> dict[str, Any]:
    return decode(token, JWT_SECRET, algorithms=[JWT_ALG])

if __name__ == '__main__':
    print(hash_password("123"))
    print(verify_password("123", "$bcrypt-sha256$v=2,t=2b,r=12$PxP3Hn.jXZ92uW9qRXzui.$79DfXhEbfYvMuYMv4pFXX6mdm.Sbxze"))

















