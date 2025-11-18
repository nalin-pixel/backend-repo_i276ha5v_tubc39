"""
Database Schemas for the Tech E‑Commerce app

Each Pydantic model corresponds to a MongoDB collection. Collection name is the lowercase of the class name.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr


class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: str = Field(..., description="Hashed password")
    avatar_url: Optional[str] = Field(None, description="Profile avatar")
    saved_build_ids: List[str] = Field(default_factory=list)


class Product(BaseModel):
    sku: str = Field(..., description="Unique stock keeping unit")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    category: str = Field(..., description="Category like 'gpu','cpu','phone','accessory','build'")
    brand: Optional[str] = None
    price: float = Field(..., ge=0)
    stock: int = Field(ge=0, default=0)
    images: List[str] = Field(default_factory=list)
    specs: Dict[str, Any] = Field(default_factory=dict)
    rating: float = Field(ge=0, le=5, default=4.5)


class PCBuild(BaseModel):
    name: str
    total_price: float
    parts: Dict[str, Dict[str, Any]]  # keys: cpu,gpu,ram,motherboard,psu,case,storage,cooler
    target_use: Optional[str] = None
    fps_estimates: Dict[str, int] = Field(default_factory=dict)
    user_id: Optional[str] = None


class Order(BaseModel):
    user_id: str
    items: List[Dict[str, Any]]
    subtotal: float
    tax: float
    shipping: float
    total: float
    status: str = Field(default="processing")


class Review(BaseModel):
    product_id: str
    user_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
