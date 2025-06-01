from typing import Any, Dict

from atria_core.schemas.base import BaseDatabaseSchema
from atria_core.schemas.utils import NameStr, SerializableUUID
from gotrue import User, UserAttributes  # type: ignore
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# Shared properties
class Token(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None


# request
class UserIn(Token):  # type: ignore
    model_config = ConfigDict(from_attributes=True, extra="ignore")
    id: SerializableUUID
    email: str | None = None
    app_metadata: Dict[str, Any] | None = None
    user_metadata: Dict[str, Any] | None = None
    aud: str | None = None


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
    full_name: str = Field(
        None, min_length=1, max_length=100, description="Full name of the user"
    )
    email: EmailStr = Field(
        None, max_length=255, description="Email address of the user"
    )
    bio: str | None = Field(
        None, max_length=500, description="Short biography of the user"
    )
    location: str | None = Field(
        None, max_length=100, description="Location of the user"
    )
    avatar_url: str | None = Field(
        None,
        description="URL of the user's avatar image",
    )
    website: str | None = Field(
        None,
        max_length=255,
        description="URL of the user's personal or professional website",
    )


class UserProfileCreate(UserProfileBase):
    pass


class UserProfileUpdate(BaseModel):
    full_name: str | None = Field(
        None, min_length=1, max_length=100, description="Full name of the user"
    )
    bio: str | None = Field(
        None, max_length=500, description="Short biography of the user"
    )
    location: str | None = Field(
        None, max_length=100, description="Location of the user"
    )
    website: str | None = Field(
        None,
        max_length=255,
        description="URL of the user's personal or professional website",
    )

    @field_validator("bio", mode="after")
    @classmethod
    def set_default_bio(cls, v):
        if not v or v.strip() == "":
            return "Hey There"
        return v


class UserProfile(UserProfileBase, BaseDatabaseSchema):
    user_id: SerializableUUID
