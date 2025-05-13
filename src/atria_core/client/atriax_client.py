from typing import Dict, Optional

import httpx
from atria_core import rest
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
from pydantic import BaseModel, EmailStr
from supabase import Client, ClientOptions, create_client

logger = get_logger(__name__)


class AuthLoginModel(BaseModel):
    email: EmailStr
    password: str


class AtriaXClient:
    def __init__(
        self,
        base_url: str,
        anon_api_key: str,
        credentials: Optional[AuthLoginModel] = None,
        initialize_auth: bool = True,
        service_name: str = "atria",
    ):
        self._service_name = service_name
        self._anon_api_key = anon_api_key
        self._credentials = credentials
        self._rest_client = httpx.Client(base_url=base_url)
        self._supabase: Client = self._init_supabase_client(
            base_url=base_url, anon_api_key=anon_api_key
        )

        self._perform_health_check()
        if initialize_auth:
            self.initialize_auth()

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
        return rest.experiment(client=self._rest_client)

    @property
    def run(self) -> RESTRun:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.run(client=self._rest_client)

    @property
    def param(self) -> RESTParam:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.param(client=self._rest_client)

    @property
    def metric(self) -> RESTMetric:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.metric(client=self._rest_client)

    # dataset
    @property
    def dataset(self) -> RESTDataset:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.dataset(client=self._rest_client)

    @property
    def dataset_version(self) -> RESTDatasetVersion:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.dataset_version(client=self._rest_client)

    @property
    def dataset_split(self) -> RESTDatasetSplit:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.dataset_split(client=self._rest_client)

    @property
    def shard_file(self) -> RESTShardFile:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.shard_file(client=self._rest_client)

    # model
    @property
    def model(self) -> RESTModel:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.model(client=self._rest_client)

    @property
    def model_version(self) -> RESTModelVersion:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.model_version(client=self._rest_client)

    # config
    @property
    def config(self) -> RESTConfig:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.config(client=self._rest_client)

    # data instance
    @property
    def document_instance(self) -> RESTDataset:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.document_instance(client=self._rest_client)

    @property
    def image_instance(self) -> RESTDataset:
        self._rest_client.headers.update(self._get_auth_headers())
        return rest.image_instance(client=self._rest_client)

    def get_session(self):
        try:
            return self._supabase.auth.get_session()
        except Exception as e:
            return None

    def _init_supabase_client(self, base_url: str, anon_api_key: str) -> Client:
        """Initialize the Supabase client (sync version)."""
        return create_client(
            supabase_url=base_url,
            supabase_key=anon_api_key,
            options=ClientOptions(storage=KeyringStorage(self._service_name)),
        )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Return headers using the Supabase session token."""
        session = self.get_session()
        if not session:
            raise RuntimeError("No active session. Please authenticate.")
        return {
            "apiKey": self._anon_api_key,
            "Authorization": f"Bearer {session.access_token}",
        }

    def _perform_health_check(self):
        """Check if the backend is healthy."""
        try:
            response = self._rest_client.get(
                "/rest/v1/utils/health-check/",
                headers={
                    "apiKey": self._anon_api_key,
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to connect to atriax server: {response.status_code} - {response.text}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to atriax server")

    def initialize_auth(self):
        """Authenticate user and store the session."""
        session = self.get_session()
        if not session:
            if self._credentials is None:
                email = input("Enter your email: ")
                password = input("Enter your password: ")
                self._credentials = AuthLoginModel(email=email, password=password)

            self.sign_in()

    def sign_in(self):
        """Sign in the user."""
        try:
            result = self._supabase.auth.sign_in_with_password(
                self._credentials.model_dump()
            )
            if not result.session or not result.user:
                raise RuntimeError("Sign-in failed. Please check your credentials.")
            logger.info(f"Sign-in successful for {self._credentials.email}")
        except Exception as e:
            logger.error(f"Failed to sign in: {e}")

    def sign_up(self, email: str, password: str, username: str):
        """Sign up a new user."""
        try:
            result = self._supabase.auth.sign_up(
                dict(
                    email=email,
                    password=password,
                    options=dict(
                        data=dict(
                            username=username,
                        )
                    ),
                ),
            )
            if not result.session or not result.user:
                raise RuntimeError("Sign-up failed. Please check your credentials.")
            logger.info(f"Sign-up successful for {email}")
        except Exception as e:
            logger.error(f"Failed to sign up: {e}")

    def sign_out(self):
        """Sign out the user by removing the session."""
        try:
            self._supabase.auth.sign_out()
            logger.info("Signed out successfully")
        except Exception as e:
            logger.error(f"Failed to sign out: {e}")
