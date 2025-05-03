from typing import Optional

import keyring
from gotrue._sync.storage import SyncSupportedStorage

from atria_core.client.config import settings
from atria_core.logger.logger import get_logger

logger = get_logger(__name__)


class KeyringStorage(SyncSupportedStorage):
    def get_item(self, key: str) -> Optional[str]:
        """Retrieve an item from keyring storage asynchronously."""
        try:
            return keyring.get_password(settings.SERVICE_NAME, key)
        except Exception as e:
            logger.error(f"Failed to get item from keyring: {e}")
            return None

    def set_item(self, key: str, value: str) -> None:
        """Set an item in keyring storage asynchronously."""
        try:
            keyring.set_password(settings.SERVICE_NAME, key, value)
        except Exception as e:
            logger.error(f"Failed to set item in keyring: {e}")

    def remove_item(self, key: str) -> None:
        """Remove an item from keyring storage asynchronously."""
        try:
            if keyring.get_password(settings.SERVICE_NAME, key) is not None:
                keyring.delete_password(settings.SERVICE_NAME, key)
        except Exception as e:
            logger.error(f"Failed to remove item from keyring: {e}")
