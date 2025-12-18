from typing import List

from pydantic import BaseModel


class SetRolePermsReq(BaseModel):
    prem_codes: List[str]

class SetUserRolesReq(BaseModel):
    role_codes: List[str]
