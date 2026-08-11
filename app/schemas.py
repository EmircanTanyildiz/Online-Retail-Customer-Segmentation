from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Müşteri yaşı")
    annual_income: float = Field(..., ge=0, description="Yıllık gelir")
    months_active: int = Field(..., ge=0, le=120, description="Aktif ay sayısı")
    avg_monthly_spend: float = Field(..., ge=0, description="Aylık ortalama harcama")
    purchase_frequency: float = Field(..., ge=0, description="Satın alma sıklığı")
    avg_order_value: float = Field(..., ge=0, description="Ortalama sipariş tutarı")
    discount_usage_rate: float = Field(..., ge=0, le=1, description="İndirim kullanım oranı")
    return_rate: float = Field(..., ge=0, le=1, description="İade oranı")
    browsing_time_minutes: float = Field(..., ge=0, description="Gezinme süresi (dk)")
    support_interactions: float = Field(..., ge=0, description="Destek etkileşim sayısı")
    payment_method: str = Field(..., description="Ödeme yöntemi")
    region: str = Field(..., description="Bölge")


class PredictionResponse(BaseModel):
    segment: str
    confidence: float
    probabilities: dict[str, float]
    segment_info: dict[str, str]
