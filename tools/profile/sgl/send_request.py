import asyncio
import os

import aiohttp

base_url = os.environ["BASE_URL"]
headers = {"Authorization": os.environ["API_KEY"], "Content-Type": "application/json"}


async def send_http_request():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/generate",
            headers=headers,
            json={
                "input_ids": [2] * 2000,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 1500,
                    "ignore_eos": True,
                },
                "stream": True,
            },
        ) as res:
            if res.status == 200:
                # dummpy process streaming content
                async for _ in res.content:
                    pass


if __name__ == "__main__":
    asyncio.run(send_http_request())
