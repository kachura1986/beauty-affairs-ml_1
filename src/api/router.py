from fastapi import APIRouter

from api.endpoints import health_check

main_router = APIRouter(prefix='/api')

# Define routers
routers = [
    # Health
    (health_check.router, '/health_check', 'Health'),
]

for current_router, prefix, tag in routers:
    main_router.include_router(current_router, prefix=prefix, tags=[tag])
