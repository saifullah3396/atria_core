from typing import Dict

import httpx
from pydantic import BaseModel, EmailStr
from supabase import Client, ClientOptions, create_client

from atria_core import rest
from atria_core.client.config import settings
from atria_core.client.keyring import KeyringStorage
from atria_core.logger.logger import get_logger
from atria_core.rest.config_rest import RESTConfig
from atria_core.rest.dataset.dataset_rest import (
    RESTDataset,
    RESTDatasetSplit,
    RESTDatasetVersion,
    RESTShardFile,
)
from atria_core.rest.model.model_rest import RESTModel, RESTModelVersion
from atria_core.rest.tracking.experiment_rest import RESTExperiment
from atria_core.rest.tracking.metric_rest import RESTMetric
from atria_core.rest.tracking.param_rest import RESTParam
from atria_core.rest.tracking.run_rest import RESTRun
from atria_core.schemas.auth import UserIn

logger = get_logger(__name__)


class AuthLoginModel(BaseModel):
    email: EmailStr
    password: str


class AtriaXClient:
    def __init__(self, auth_enabled: bool = True):
        self._supabase: Client = self._init_supabase_client()
        self._rest_client = httpx.Client(base_url=settings.ATRIAX_URL)
        self._auth_enabled = auth_enabled

        self._perform_health_check()
        if self._auth_enabled:
            self._perform_auth()

    @property
    def rest_client(self) -> httpx.Client:
        """Return the HTTP client for REST API calls."""
        return self._rest_client

    @property
    def supabase_client(self) -> Client:
        """Return the Supabase client."""
        return self._supabase

    @property
    def user(self) -> UserIn:
        """Return the user ID from the Supabase session."""
        session = self.get_session()
        if not session:
            raise RuntimeError("No active session. Please authenticate.")
        return session.user

    # tracking
    @property
    def experiment(self) -> RESTExperiment:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.experiment(self._rest_client)

    @property
    def run(self) -> RESTRun:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.run(self._rest_client)

    @property
    def param(self) -> RESTParam:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.param(self._rest_client)

    @property
    def metric(self) -> RESTMetric:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.metric(self._rest_client)

    # dataset
    @property
    def dataset(self) -> RESTDataset:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.dataset(self._rest_client)

    @property
    def dataset_version(self) -> RESTDatasetVersion:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.dataset_version(self._rest_client)

    @property
    def dataset_split(self) -> RESTDatasetSplit:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.dataset_split(self._rest_client)

    @property
    def shard_file(self) -> RESTShardFile:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.shard_file(self._rest_client)

    # model
    @property
    def model(self) -> RESTModel:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.model(self._rest_client)

    @property
    def model_version(self) -> RESTModelVersion:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.model_version(self._rest_client)

    # config
    @property
    def config(self) -> RESTConfig:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.config(self._rest_client)

    # data instance
    @property
    def document_instance(self) -> RESTDataset:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.document_instance(self._rest_client)

    @property
    def image_instance(self) -> RESTDataset:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.image_instance(self._rest_client)

    def get_session(self):
        try:
            return self._supabase.auth.get_session()
        except Exception as e:
            return None

    def _init_supabase_client(self) -> Client:
        """Initialize the Supabase client (sync version)."""
        return create_client(
            settings.ATRIAX_URL,
            settings.ATRIAX_API_KEY,
            options=ClientOptions(storage=KeyringStorage()),
        )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Return headers using the Supabase session token."""
        session = self.get_session()
        if not session:
            raise RuntimeError("No active session. Please authenticate.")
        return {
            "apiKey": settings.ATRIAX_API_KEY,
            "Authorization": f"Bearer {session.access_token}",
        }

    def _perform_health_check(self):
        """Check if the backend is healthy."""
        try:
            response = self._rest_client.get(
                "/rest/v1/utils/health-check/",
                headers={
                    "apiKey": settings.ATRIAX_API_KEY,
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to connect to atriax server: {response.status_code} - {response.text}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to atriax server")

    def _perform_auth(self):
        """Authenticate user and store the session."""
        session = self.get_session()
        if not session:
            email = input("Enter your email: ")
            password = input("Enter your password: ")
            login_data = AuthLoginModel(email=email, password=password)

            result = self._supabase.auth.sign_in_with_password(
                {"email": login_data.email, "password": login_data.password}
            )
            if not result.session or not result.user:
                raise RuntimeError(
                    "Authentication failed. Please check your credentials."
                )
