from gotrue import User, UserAttributes  # type: ignore
from pydantic import BaseModel, EmailStr, Field

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import NameStr, SerializableUUID


# Shared properties
class Token(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None


# request
class UserIn(Token, User):  # type: ignore
    id: SerializableUUID


# Properties to receive via API on creation
# in
class UserCreate(BaseModel):
    username: str


# Properties to receive via API on update
# in
class UserUpdate(UserAttributes):  # type: ignore
    pass


# response


class UserInDBBase(BaseModel):
    pass


# Properties to return to client via api
# out
class UserOut(Token):
    pass


# Properties properties stored in DB
class UserInDB(User):  # type: ignore
    pass


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserProfileBase(BaseModel):
    username: NameStr = Field(..., min_length=1, max_length=50)


class UserProfileCreate(UserProfileBase):
    user_id: SerializableUUID


class UserProfileUpdate(UserProfileBase, OptionalModel):
    pass


class UserProfile(UserProfileBase, BaseDatabaseSchema):
    user_id: SerializableUUID
