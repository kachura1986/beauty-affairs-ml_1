from fastapi import APIRouter

from api.endpoints import agent, health_check

main_router = APIRouter(prefix='/api')

# Define routers
routers = [
    # Health
    (health_check.router, '/health_check', 'Health'),

    # Generate
    (agent.router, '/agent', ['agent']),
]

for current_router, prefix, tag in routers:
    main_router.include_router(current_router, prefix=prefix, tags=[tag])
