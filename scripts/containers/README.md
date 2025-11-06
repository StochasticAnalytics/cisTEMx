Note on pushing locally, you have to create a *Classic* personal access token as peronal user -> dev settings -> generate. Then copy token, then

```bash
echo <your token here> | docker login ghcr.io -u <gh username here> --password-stdin
```

v. annoying. Prefer just CI/CD (which is a pain for developing large changes to the docker file.)
