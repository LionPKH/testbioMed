from pydantic import BaseModel
from typing import Optional
from datetime import date

class RegisterUser(BaseModel):
    username: str
    email: str
    password: str
    bio: Optional[str] = None
    birth_date: Optional[date] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    subscription_active: Optional[bool] = False

class RegisterAdmin(BaseModel):
    username: str
    email: str
    password: str
    department: Optional[str] = None
    phone: Optional[str] = None
    permissions_level: Optional[int] = 1
    access_code: Optional[str] = "default"

class UpdateUser(BaseModel):
    username: Optional[str]
    email: Optional[str]
    user_type: Optional[str]
    password: Optional[str]
    bio: Optional[str]
    birth_date: Optional[date]
    phone: Optional[str]
    city: Optional[str]
    subscription_active: Optional[bool]
    department: Optional[str]
    permissions_level: Optional[int]
    access_code: Optional[str]
