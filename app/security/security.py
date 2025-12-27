# 安全工具

from datetime import timezone, datetime, timedelta
from typing import Any
from app.config import settings
from jwt import encode, decode
from passlib.context import CryptContext

PWD_CONTEXT = CryptContext(
    schemes=["argon2", "bcrypt_sha256", "pbkdf2_sha256"],  # 支持的算法列表，按偏好排序
    default="bcrypt_sha256",  # 创建新哈希时，默认使用 argon2
    deprecated="auto"
)

def hash_password(password) -> str:
    return PWD_CONTEXT.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return PWD_CONTEXT.verify(plain_password, hashed_password)

def create_access_token(payload: dict[str, Any], expires_minutes: int | None = None) -> str:
    minutes = expires_minutes or settings.JWT_EXPIRE_MINUTES
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    to_encode = {**payload, "exp": exp}
    return encode(to_encode, settings.JWT_SECRET, settings.JWT_ALG)

def decode_token(token: str) -> dict[str, Any]:
    return decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])

if __name__ == '__main__':

    print(hash_password("123"))
    print(verify_password("123", "$bcrypt-sha256$v=2,t=2b,r=12$PxP3Hn.jXZ92uW9qRXzui.$79DfXhEbfYvMuYMv4pFXX6mdm.Sbxze"))

















