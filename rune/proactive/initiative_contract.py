"""Initiative contract management for RUNE.

Tracks the lifecycle of proactive initiatives from creation through
delivery to resolution (accepted/dismissed/expired).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)

ContractStatus = Literal["pending", "delivered", "accepted", "dismissed", "expired"]

# Default contract TTL in seconds
_DEFAULT_TTL = 600  # 10 minutes


@dataclass(slots=True)
class InitiativeContract:
    """Represents a proactive initiative and its lifecycle state."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    trigger_type: str = ""
    suggestion: Suggestion = field(default_factory=Suggestion)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(seconds=_DEFAULT_TTL)
    )
    status: ContractStatus = "pending"


class ContractManager:
    """Manages the lifecycle of initiative contracts.

    Contracts transition through: pending -> delivered -> accepted/dismissed.
    Expired contracts are cleaned up periodically.
    """

    __slots__ = ("_contracts",)

    def __init__(self) -> None:
        self._contracts: dict[str, InitiativeContract] = {}

    def create(
        self,
        trigger_type: str,
        suggestion: Suggestion,
        ttl_seconds: float = _DEFAULT_TTL,
    ) -> InitiativeContract:
        """Create a new initiative contract.

        Parameters:
            trigger_type: What triggered this initiative (e.g., "idle", "time_trigger").
            suggestion: The proactive suggestion associated with this contract.
            ttl_seconds: Time-to-live before the contract expires.

        Returns:
            The newly created contract.
        """
        now = datetime.now()
        contract = InitiativeContract(
            trigger_type=trigger_type,
            suggestion=suggestion,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            status="pending",
        )
        self._contracts[contract.id] = contract

        log.debug(
            "contract_created",
            contract_id=contract.id,
            trigger=trigger_type,
            suggestion_type=suggestion.type,
        )
        return contract

    def deliver(self, contract_id: str) -> None:
        """Mark a contract as delivered to the user.

        Parameters:
            contract_id: The ID of the contract to mark.
        """
        contract = self._contracts.get(contract_id)
        if contract is None:
            log.warning("contract_not_found", contract_id=contract_id, action="deliver")
            return

        if contract.status == "pending":
            contract.status = "delivered"
            log.debug("contract_delivered", contract_id=contract_id)

    def resolve(self, contract_id: str, accepted: bool) -> None:
        """Resolve a contract as accepted or dismissed.

        Parameters:
            contract_id: The ID of the contract to resolve.
            accepted: Whether the user accepted the suggestion.
        """
        contract = self._contracts.get(contract_id)
        if contract is None:
            log.warning("contract_not_found", contract_id=contract_id, action="resolve")
            return

        contract.status = "accepted" if accepted else "dismissed"
        log.debug(
            "contract_resolved",
            contract_id=contract_id,
            accepted=accepted,
        )

    def cleanup_expired(self) -> int:
        """Mark and remove expired contracts.

        Returns:
            The number of contracts cleaned up.
        """
        now = datetime.now()
        expired_ids: list[str] = []

        for cid, contract in self._contracts.items():
            if contract.status in ("pending", "delivered") and contract.expires_at < now:
                contract.status = "expired"
                expired_ids.append(cid)

        for cid in expired_ids:
            del self._contracts[cid]

        if expired_ids:
            log.debug("contracts_expired", count=len(expired_ids))

        return len(expired_ids)

    def get_active(self) -> list[InitiativeContract]:
        """Return all non-expired, non-resolved contracts.

        Returns:
            List of active contracts (pending or delivered).
        """
        now = datetime.now()
        return [
            c
            for c in self._contracts.values()
            if c.status in ("pending", "delivered") and c.expires_at >= now
        ]
