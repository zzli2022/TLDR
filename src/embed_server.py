

import numpy as np

from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import numpy as np
import transformers
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import json
import ssl
from typing import AsyncGenerator
import torch
import argparse
import transformers

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

def load_model(model_args):
    model = SentenceTransformer(model_args.model_name_or_path, 
                                trust_remote_code=True)
    model = model.eval()
    return model
    
device = 'cuda:0'

def infer_embed(prompt):
    if isinstance(prompt, str):
        prompt = [prompt]
    
    embeddings_ = model.encode(prompt)
    embeddings_ = embeddings_ / (norm(embeddings_, axis=-1, keepdims=True)+1e-20)
    embeddings_list = embeddings_.tolist()
    output_dict = []
    for p, embed in zip(prompt, embeddings_list):
        tmp = {
            'text': p,
            'embed': embed
        }
        output_dict.append(tmp)
    return output_dict

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/get_embed")
async def get_embed(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    
    output_dict = infer_embed(prompt)
    return JSONResponse(output_dict)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=None)
    
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    
    parser.add_argument("--log-level", type=str, default="debug")
    args = parser.parse_args()
    
    model = load_model(args)
    model = model.to(device)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)