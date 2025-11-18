import os
import hashlib
import random
import string
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents

app = FastAPI(title="Tech Commerce API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Utilities ----------------------

def oid_to_str(d: Dict[str, Any]) -> Dict[str, Any]:
    if d is None:
        return d
    d = dict(d)
    if d.get("_id") is not None:
        d["id"] = str(d.pop("_id"))
    return d


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def gen_token(n: int = 32) -> str:
    return "tok_" + "".join(random.choices(string.ascii_letters + string.digits, k=n))


# ---------------------- Seed Data ----------------------

SAMPLE_PRODUCTS = [
    {
        "sku": "CPU-5600X",
        "title": "AMD Ryzen 5 5600X",
        "category": "cpu",
        "brand": "AMD",
        "price": 159.99,
        "stock": 25,
        "images": [
            "https://images.unsplash.com/photo-1612198185721-5f0076f0e5b2?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"socket": "AM4", "cores": 6, "threads": 12, "base_clock": 3.7},
        "rating": 4.7,
        "description": "Zen 3 architecture with excellent gaming performance",
    },
    {
        "sku": "GPU-3060",
        "title": "NVIDIA GeForce RTX 3060",
        "category": "gpu",
        "brand": "NVIDIA",
        "price": 279.0,
        "stock": 14,
        "images": [
            "https://images.unsplash.com/photo-1600861194942-f883de0dfe96?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"vram_gb": 12, "length_mm": 242, "tdp_w": 170, "pcie": 4},
        "rating": 4.6,
        "description": "Great 1080p/1440p value with RTX features",
    },
    {
        "sku": "MB-B550",
        "title": "MSI B550 Tomahawk",
        "category": "motherboard",
        "brand": "MSI",
        "price": 149.0,
        "stock": 8,
        "images": [
            "https://images.unsplash.com/photo-1555617981-dac3880f7f3b?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"socket": "AM4", "chipset": "B550", "size": "ATX", "pcie": 4, "m2_slots": 2},
        "rating": 4.5,
    },
    {
        "sku": "RAM-16-3200",
        "title": "Corsair Vengeance LPX 16GB (2x8) 3200MHz DDR4",
        "category": "ram",
        "brand": "Corsair",
        "price": 49.99,
        "stock": 50,
        "images": [
            "https://images.unsplash.com/photo-1615869442320-04bc14b1a157?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"type": "DDR4", "capacity_gb": 16, "speed_mhz": 3200},
        "rating": 4.8,
    },
    {
        "sku": "CASE-NZXT-H510",
        "title": "NZXT H510",
        "category": "case",
        "brand": "NZXT",
        "price": 79.0,
        "stock": 12,
        "images": [
            "https://images.unsplash.com/photo-1587202372775-98927b65f555?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"size": "ATX", "gpu_clearance_mm": 381},
        "rating": 4.4,
    },
    {
        "sku": "PSU-650",
        "title": "Corsair RM650 650W 80+ Gold",
        "category": "psu",
        "brand": "Corsair",
        "price": 89.0,
        "stock": 22,
        "images": [
            "https://images.unsplash.com/photo-1618498082410-b4b0a3e5baa0?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"wattage": 650},
        "rating": 4.6,
    },
    {
        "sku": "SSD-1TB",
        "title": "Samsung 980 1TB NVMe",
        "category": "storage",
        "brand": "Samsung",
        "price": 79.0,
        "stock": 30,
        "images": [
            "https://images.unsplash.com/photo-1606811971592-7f1e4b1cd5df?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"type": "NVMe", "size_gb": 1000, "pcie": 3},
        "rating": 4.7,
    },
    {
        "sku": "PHONE-14",
        "title": "Apple iPhone 14 128GB",
        "category": "phone",
        "brand": "Apple",
        "price": 699.0,
        "stock": 18,
        "images": [
            "https://images.unsplash.com/photo-1668554245160-08bda8a0ffdc?q=80&w=1200&auto=format&fit=crop"
        ],
        "specs": {"os": "iOS", "storage_gb": 128},
        "rating": 4.9,
    },
]

FPS_DB = {
    "cpu_score": {"AMD Ryzen 5 5600X": 140, "default": 100},
    "gpu_score": {"NVIDIA GeForce RTX 3060": 140, "default": 100},
    "games": {
        "Fortnite": 1.0,
        "Valorant": 1.2,
        "GTA V": 0.9,
        "RDR2": 0.7,
        "Cyberpunk": 0.6,
    },
}


def seed_products():
    if db is None:
        return
    count = db["product"].count_documents({})
    if count == 0:
        for p in SAMPLE_PRODUCTS:
            create_document("product", p)


seed_products()

# ---------------------- Models ----------------------

class RegisterPayload(BaseModel):
    name: str
    email: str
    password: str


class LoginPayload(BaseModel):
    email: str
    password: str


class BuilderGamesPayload(BaseModel):
    games: List[str]
    budget: Optional[float] = None


class ComponentsPayload(BaseModel):
    cpu: Optional[Dict[str, Any]] = None
    gpu: Optional[Dict[str, Any]] = None
    ram: Optional[Dict[str, Any]] = None
    motherboard: Optional[Dict[str, Any]] = None
    case: Optional[Dict[str, Any]] = None
    psu: Optional[Dict[str, Any]] = None
    storage: Optional[Dict[str, Any]] = None
    cooler: Optional[Dict[str, Any]] = None


class ChatPayload(BaseModel):
    message: str
    budget: Optional[float] = None


# ---------------------- Basic routes ----------------------

@app.get("/")
def root():
    return {"ok": True, "name": "Tech Commerce API"}


@app.get("/schema")
def get_schema():
    # Expose pydantic schemas file for tooling
    try:
        from schemas import User, Product, PCBuild, Order, Review
        return {
            "collections": [
                {"name": "user", "schema": User.model_json_schema()},
                {"name": "product", "schema": Product.model_json_schema()},
                {"name": "pcbuild", "schema": PCBuild.model_json_schema()},
                {"name": "order", "schema": Order.model_json_schema()},
                {"name": "review", "schema": Review.model_json_schema()},
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------- Auth ----------------------

@app.post("/api/auth/register")
def register(payload: RegisterPayload):
    if db is None:
        raise HTTPException(500, "Database not configured")
    existing = db["user"].find_one({"email": payload.email})
    if existing:
        raise HTTPException(400, "Email already registered")
    doc = {
        "name": payload.name,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "saved_build_ids": [],
    }
    user_id = create_document("user", doc)
    return {"user_id": user_id, "token": gen_token()}


@app.post("/api/auth/login")
def login(payload: LoginPayload):
    if db is None:
        raise HTTPException(500, "Database not configured")
    user = db["user"].find_one({"email": payload.email})
    if not user:
        raise HTTPException(401, "Invalid credentials")
    if user.get("password_hash") != hash_password(payload.password):
        raise HTTPException(401, "Invalid credentials")
    return {"user_id": str(user["_id"]), "token": gen_token()}


# ---------------------- Products ----------------------

@app.get("/api/products")
def list_products(
    q: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    in_stock: Optional[bool] = None,
    sort: Optional[str] = Query(None, description="price_asc|price_desc|newest|popular"),
    limit: int = 50,
):
    if db is None:
        raise HTTPException(500, "Database not configured")

    filt: Dict[str, Any] = {}
    if q:
        filt["$or"] = [{"title": {"$regex": q, "$options": "i"}}, {"description": {"$regex": q, "$options": "i"}}]
    if category:
        filt["category"] = category
    if brand:
        filt["brand"] = brand
    if in_stock is not None:
        filt["stock" if in_stock else "$or"] = ( {"$gt": 0} if in_stock else [{"stock": 0}] )

    cursor = db["product"].find(filt)
    if sort == "price_asc":
        cursor = cursor.sort("price", 1)
    elif sort == "price_desc":
        cursor = cursor.sort("price", -1)
    elif sort == "newest":
        cursor = cursor.sort("created_at", -1)

    items = [oid_to_str(p) for p in cursor.limit(limit)]
    return {"items": items}


@app.get("/api/products/{product_id}")
def product_detail(product_id: str):
    if db is None:
        raise HTTPException(500, "Database not configured")
    try:
        obj = db["product"].find_one({"_id": ObjectId(product_id)})
    except Exception:
        obj = db["product"].find_one({"sku": product_id})
    if not obj:
        raise HTTPException(404, "Product not found")
    return oid_to_str(obj)


# ---------------------- PC Builder ----------------------

@app.post("/api/builder/games")
def builder_games(payload: BuilderGamesPayload):
    games = payload.games or []
    if not games:
        raise HTTPException(400, "No games provided")

    # Use very simple heuristics for three tiers
    base_cpu = db["product"].find_one({"category": "cpu"}) or {}
    base_gpu = db["product"].find_one({"category": "gpu"}) or {}
    base_mb = db["product"].find_one({"category": "motherboard"}) or {}
    base_ram = db["product"].find_one({"category": "ram"}) or {}
    base_case = db["product"].find_one({"category": "case"}) or {}
    base_psu = db["product"].find_one({"category": "psu"}) or {}
    base_storage = db["product"].find_one({"category": "storage"}) or {}

    def build(mult: float):
        # scale prices for tiers
        parts = [base_cpu, base_gpu, base_mb, base_ram, base_case, base_psu, base_storage]
        chosen = []
        total = 0.0
        for p in parts:
            if not p:
                continue
            price = float(p.get("price", 0)) * mult
            total += price
            chosen.append({"id": str(p.get("_id", "")), "title": p.get("title"), "price": round(price, 2), "category": p.get("category")})
        # FPS: combine CPU/GPU scores * game multipliers
        cpu_score = FPS_DB["cpu_score"].get(base_cpu.get("title", ""), FPS_DB["cpu_score"]["default"]) * mult
        gpu_score = FPS_DB["gpu_score"].get(base_gpu.get("title", ""), FPS_DB["gpu_score"]["default"]) * mult
        fps = {g: int((cpu_score * 0.4 + gpu_score * 0.6) * FPS_DB["games"].get(g, 1.0)) for g in games}
        return {"parts": chosen, "total_price": round(total, 2), "fps": fps}

    cheapest = build(0.8)
    balanced = build(1.0)
    best = build(1.4)

    return {"cheapest": cheapest, "balanced": balanced, "best": best}


@app.post("/api/builder/components/validate")
def builder_components(payload: ComponentsPayload):
    comp = payload.model_dump()
    notes: List[str] = []

    # Compatibility checks (simplified)
    cpu = comp.get("cpu") or {}
    mb = comp.get("motherboard") or {}
    if cpu and mb:
        if cpu.get("specs", {}).get("socket") != mb.get("specs", {}).get("socket"):
            notes.append("CPU and motherboard sockets do not match")

    ram = comp.get("ram") or {}
    if ram and mb:
        if (ram.get("specs", {}).get("type") or "").replace(" ", "") not in (mb.get("specs", {}).get("ram_type", "DDR4,DDR5")):
            # assume board supports DDR4,DDR5 unless specified
            pass

    gpu = comp.get("gpu") or {}
    case = comp.get("case") or {}
    if gpu and case:
        if gpu.get("specs", {}).get("length_mm", 0) > case.get("specs", {}).get("gpu_clearance_mm", 9999):
            notes.append("GPU may not fit in the selected case")

    psu = comp.get("psu") or {}
    if psu and gpu:
        if psu.get("specs", {}).get("wattage", 0) < gpu.get("specs", {}).get("tdp_w", 0) + 200:
            notes.append("Power supply wattage might be insufficient")

    # Simple performance estimation
    def score_of(item: Dict[str, Any], table: Dict[str, int]) -> int:
        if not item:
            return 0
        return table.get(item.get("title", ""), table.get("default", 100))

    fps_est = int((score_of(cpu, FPS_DB["cpu_score"]) * 0.4 + score_of(gpu, FPS_DB["gpu_score"]) * 0.6))

    return {"notes": notes, "estimated_fps_1080p": fps_est}


# ---------------------- AI-like Chat ----------------------

@app.post("/api/chat")
def ai_chat(payload: ChatPayload):
    text = payload.message.lower()
    budget = payload.budget or 1000

    # Simple heuristic suggestions
    suggestion = {
        "title": "Balanced 1080p Esports Build",
        "parts": [
            {"category": "cpu", "title": "AMD Ryzen 5 5600X"},
            {"category": "gpu", "title": "NVIDIA GeForce RTX 3060"},
            {"category": "ram", "title": "16GB DDR4 3200"},
            {"category": "motherboard", "title": "B550 ATX"},
            {"category": "storage", "title": "1TB NVMe"},
            {"category": "psu", "title": "650W Gold"},
            {"category": "case", "title": "NZXT H510"},
        ],
        "estimated_total": min(max(budget, 700), 1500),
        "fps_estimates": {"Valorant": 200, "Fortnite": 160},
        "alternatives": ["RTX 3060 Ti", "RX 6700 XT", "Ryzen 7 5700X"],
    }

    reply = "Here's a balanced build tuned for high FPS in esports titles."
    if "144" in text or "fps" in text:
        reply = "To hit 144 FPS consistently, this balanced build will do great at 1080p."
    if "$" in text or "budget" in text:
        reply += " I adjusted it for your budget."

    return {"message": reply, "suggestion": suggestion}


# ---------------------- Test ----------------------

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
