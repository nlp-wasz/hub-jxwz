from pydantic import BaseModel,Field
from typing import Optional,List,Dict
from typing_extensions import Literal

# 领域枚举
DOMAINS = Literal[
    'music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather',
    'match', 'map', 'website', 'news', 'message', 'contacts', 'translation',
    'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone',
    'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'
]

# 意图枚举
INTENTS = Literal[
    'OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
    'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
    'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
    'TRANSLATION', 'VIEW', 'ROUTE', 'POSITION'
]

# 实体枚举
ENTITIES = Literal[
    'code', 'Src', 'startDate_dateOrig', 'film', 'endLoc_city', 'artistRole',
    'location_country', 'location_area', 'author', 'startLoc_city', 'season',
    'dishNamet', 'media', 'datetime_date', 'episode', 'teleOperator',
    'questionWord', 'receiver', 'ingredient', 'name', 'startDate_time',
    'startDate_date', 'location_province', 'endLoc_poi', 'artist', 'dynasty',
    'area', 'location_poi', 'relIssue', 'Dest', 'content', 'keyword', 'target',
    'startLoc_area', 'tvchannel', 'type', 'song', 'queryField', 'awayName',
    'headNum', 'homeName', 'decade', 'payment', 'popularity', 'tag',
    'startLoc_poi', 'date', 'startLoc_province', 'endLoc_province',
    'location_city', 'absIssue', 'utensil', 'scoreDescr', 'dishName',
    'endLoc_area', 'resolution', 'yesterday', 'timeDescr', 'category',
    'subfocus', 'theatre', 'datetime_time'
]

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: DOMAINS = Field(description="领域")
    intent: INTENTS = Field(description="意图")
    slots: Dict[ENTITIES, str] = Field(description="实体槽位字典")

class ExtractionRequest(BaseModel):
    text: str = Field(description="待分析的文本")
    method: str = Field(default="tool", description="分析方法: tool或prompt")

class ExtractionResponse(BaseModel):
    success: bool
    data: Optional[IntentDomainNerTask] = None
    error: Optional[str] = None
    method: str