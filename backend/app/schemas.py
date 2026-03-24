from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
 
 
ChartType = Literal["bar", "line", "area", "pie"]
 
 
class Citation(BaseModel):
    page_id: str
    url: str
    title: str
    chunk_id: str
    chunk_type: str = "text"
 
 
class DataPoint(BaseModel):
    x: str
    y: float
    series: Optional[str] = None
 
 
class ChartSpec(BaseModel):
    title: str
    chart_type: ChartType
    x_label: str
    y_label: str
    units: Optional[str] = None
    data: List[DataPoint] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
 
 
class AskRequest(BaseModel):
    question: str
    want_chart: bool = True
    filters: Optional[Dict[str, Any]] = None
 
 
class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    chart: Optional[ChartSpec] = None