from typing import IO, Generator, Tuple, Union, overload
from pathlib import Path, PosixPath, PurePosixPath
import io
import os
import re
import requests

from azure.storage.blob import ContainerClient, BlobClient
import requests.adapters
import requests.packages
from urllib3.util.retry import Retry


__all__ = [
    'download_blob', 'upload_blob', 
    'download_blob_with_cache', 
    'open_blob', 'open_blob_with_cache', 
    'blob_file_exists', 
    'AzureBlobPath','SmartPath'
]


def download_blob(blob: Union[str, BlobClient]) -> bytes:
    if isinstance(blob, str):
        blob_client = BlobClient.from_blob_url(blob_client)
    else:
        blob_client = blob
    return blob_client.download_blob().read()


def upload_blob(blob: Union[str, BlobClient], data: Union[str, bytes]):
    if isinstance(blob, str):
        blob_client = BlobClient.from_blob_url(blob)
    else:
        blob_client = blob
    blob_client.upload_blob(data, overwrite=True)


def download_blob_with_cache(container: Union[str, ContainerClient], blob_name: str, cache_dir: str = 'blobcache') -> bytes:
    """
    Download a blob file from a container and return its content as bytes.
    If the file is already present in the cache, it is read from there.
    """
    cache_path = Path(cache_dir) / blob_name
    if cache_path.exists():
        return cache_path.read_bytes()
    data = download_blob(container, blob_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def open_blob(container: Union[str, ContainerClient], blob_name: str) -> io.BytesIO:
    """
    Open a blob file for reading from a container and return its content as a BytesIO object.
    """
    return io.BytesIO(download_blob(container, blob_name))


def open_blob_with_cache(container: Union[str, ContainerClient], blob_name: str, cache_dir: str = 'blobcache') -> io.BytesIO:
    """
    Open a blob file for reading from a container and return its content as a BytesIO object.
    If the file is already present in the cache, it is read from there.
    """
    return io.BytesIO(download_blob_with_cache(container, blob_name, cache_dir=cache_dir))


def blob_file_exists(container: Union[str, ContainerClient], blob_name: str) -> bool:
    """
    Check if a blob file exists in a container.
    """
    if isinstance(container, str):
        container = ContainerClient.from_container_url(container)
    blob_client = container.get_blob_client(blob_name)
    return blob_client.exists()

def is_blob_url(url: str) -> bool:
    return re.match(r'https://[^/]+blob.core.windows.net/[^/]+', url) is not None


def split_blob_url(url: str) -> Tuple[str, str, str]:
    match = re.match(r'(https://[^/]+blob.core.windows.net/[^/]+)(/([^\?]*))?(\?.+)?', url)
    if match:
        container, _, path, sas = match.groups()
        return container, path or '', sas or ''
    raise ValueError(f'Not a valid blob URL: {url}')


def join_blob_path(url: str, *others: str) -> str:
    container, path, sas = split_blob_url(url)
    return container + '/' + os.path.join(path, *others) + sas


class AzureBlobStringWriter(io.StringIO):
    def __init__(self, blob_client: BlobClient, encoding: str = 'utf-8'):
        self.encoding = encoding
        self.blob_client = blob_client
        super().__init__()

    def close(self):
        self.blob_client.upload_blob(self.getvalue().encode(self.encoding), blob_type='BlockBlob', overwrite=True)


class AzureBlobBytesWriter(io.BytesIO):
    def __init__(self, blob_client: BlobClient):
        super().__init__()
        self.blob_client = blob_client

    def close(self):
        self.blob_client.upload_blob(self.getvalue(), blob_type='BlockBlob', overwrite=True)


def open_azure_blob(blob: Union[str, BlobClient], mode: str = 'r', encoding: str = 'utf-8', newline: str = None) -> IO:
    if isinstance(blob, str):
        blob_client = BlobClient.from_blob_url(blob)
    elif isinstance(blob, BlobClient):
        blob_client = blob
    else:
        raise ValueError(f'Must be a blob URL or a BlobClient object: {blob}')
    
    if mode == 'r':
        return io.StringIO(blob_client.download_blob().read().decode(encoding), newline=newline)
    elif mode == 'rb':
        return io.BytesIO(blob_client.download_blob().read())
    elif mode == 'w':
        return AzureBlobStringWriter(blob_client)
    elif mode == 'wb':
        return AzureBlobBytesWriter(blob_client)
    else:
        raise ValueError(f'Unsupported mode: {mode}')


def smart_open(path_or_url: Union[Path, str], mode: str = 'r', encoding: str = 'utf-8') -> IO:
    if is_blob_url(str(path_or_url)):
        return open_azure_blob(str(path_or_url), mode, encoding)
    return open(path_or_url, mode, encoding)


class AzureBlobPath:
    """
    Implementation of pathlib.Path like interface for Azure Blob Storage.
    """
    container_client: ContainerClient
    path: PurePosixPath

    @overload
    def __init__(self, url: str) -> None: ...
    @overload
    def __init__(self, url: 'AzureBlobPath') -> None: ...
    @overload
    def __init__(self, *, container_client: Union[str, ContainerClient], path: Union[str, PurePosixPath]) -> None: ...
    def __init__(self, *args: Union[str, PurePosixPath], pool_maxsize: int = 256, retries: int = 3, share_client: bool = True):
        if len(args) == 1:
            if isinstance(args[0], AzureBlobPath):
                # Copy constructor
                self.container_client = args[0].container_client
                self.path = args[0].path
            elif isinstance(args[0], str):
                # Parse blob URL
                url = args[0]
                container, path, sas = split_blob_url(url)
                session = self._get_session(pool_maxsize=pool_maxsize, retries=retries)
                self.container_client = ContainerClient.from_container_url(container + sas, session=session)
                self.path = PurePosixPath(path)
        else:
            # Construct from container client and path
            self.container_client, self.path = args
            if isinstance(self.container_client, str):
                session = self._get_session(pool_maxsize=pool_maxsize, retries=retries)
                self.container_client = ContainerClient.from_container_url(self.container_client, session=session)
            if isinstance(self.path, str):
                self.path = PurePosixPath(self.path)
        self.share_client = share_client
    
    def _get_session(self, pool_maxsize: int = 1024, retries: int = 3) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE"],
            backoff_factor=1,
            raise_on_status=False,
            read=retries,
            connect=retries,
            redirect=retries,
        )
        adapter = requests.adapters.HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @property
    def url(self) -> str:
        return join_blob_path(self.container_client.url, self.path.as_posix())

    @property
    def container(self) -> str:
        return self.container_client.url
    
    def __str__(self):
        return self.url
    
    def __repr__(self):
        return self.url
    
    def open(self, mode: str = 'r', encoding: str = 'utf-8') -> IO:
        blob_client = self.container_client.get_blob_client(self.path.as_posix())
        return open_azure_blob(blob_client, mode, encoding)

    def joinpath(self, *other: str) -> 'AzureBlobPath':
        "Return a new AzureBlobPath with the path joined with other paths."
        return AzureBlobPath(self.container_client if self.share_client else self.container_client.url, self.path.joinpath(*other))

    def __truediv__(self, other: Union[str, Path]) -> 'AzureBlobPath':
        return self.joinpath(other)

    def mkdir(self, parents: bool = False, exist_ok: bool = False):
        pass

    @property
    def name(self) -> str:
        return self.path.name
    
    @property
    def parent(self) -> 'AzureBlobPath':
        return AzureBlobPath(self.container_client, self.path.parent)
    
    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def suffix(self) -> str:
        return self.path.suffix

    def with_suffix(self, suffix: str) -> 'AzureBlobPath':
        return AzureBlobPath(self.container_client, self.path.with_suffix(suffix))

    def iterdir(self, share_client: bool = True) -> Generator['AzureBlobPath', None, None]:
        path = self.path.as_posix()
        if not path.endswith('/'):
            path += '/'
        for item in self.container_client.walk_blobs(path):
            yield AzureBlobPath(self.container_client if share_client else self.container_client.url, item.name)

    def exists(self) -> bool:
        return self.container_client.get_blob_client(self.path.as_posix()).exists()

    def read_bytes(self) -> bytes:
        blob_client = self.container_client.get_blob_client(self.path.as_posix())
        return blob_client.download_blob().readall()

    def read_text(self, encoding: str = 'utf-8') -> str:
        blob_client = self.container_client.get_blob_client(self.path.as_posix())
        return blob_client.download_blob().readall().decode(encoding)
    
    def write_bytes(self, data: bytes):
        blob_client = self.container_client.get_blob_client(self.path.as_posix())
        blob_client.upload_blob(data, overwrite=True)
    
    def write_text(self, data: str, encoding: str = 'utf-8'):
        blob_client = self.container_client.get_blob_client(self.path.as_posix())
        blob_client.upload_blob(data.encode(encoding), overwrite=True)

    def unlink(self):
        blob_client = self.container_client.get_blob_client(self.path.as_posix())
        blob_client.delete_blob()


class SmartPath(Path, AzureBlobPath):
    """
    Supports both local file paths and Azure Blob Storage URLs.
    """
    def __new__(cls, path_or_url: Union[Path, str]) -> Union[Path, AzureBlobPath]:
        if is_blob_url(str(path_or_url)):
            return AzureBlobPath(str(path_or_url))
        return Path(path_or_url)
