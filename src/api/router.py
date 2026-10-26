from fastapi import APIRouter

from api.endpoints import agent, health_check, products_and_services

main_router = APIRouter(prefix="/api")

# Define routers
routers = [
    # Health
    (health_check.router, "/health-check", "Health"),

    # Generate
    (agent.router, "/agent", "Agent"),

    # Products and services
    (products_and_services.router, "/products-and-services", "Products and services"),
]

for current_router, prefix, tag in routers:
    main_router.include_router(current_router, prefix=prefix, tags=[tag])
