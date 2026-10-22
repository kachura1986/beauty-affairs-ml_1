import asyncio

from dependencies import PipelineDependencies
from services.search_index.products_search_index import ProductsSearchIndex
from services.search_index.services_search_index import ServicesSearchIndex


async def main() -> None:
    """
    Creates product and service indexes.
    """
    dependencies = await PipelineDependencies.create()
    azure_search_manager = dependencies.azure_search_index_manager

    products_index_instance = ProductsSearchIndex(
        index_name='products_test',
        azure_search_index_manager=azure_search_manager,
        embedder=dependencies.azure_embedder
    )

    services_index_instance = ServicesSearchIndex(
        index_name='services_test',
        azure_search_index_manager=azure_search_manager,
        embedder=dependencies.azure_embedder
    )

    products = [
        {
            "id": "f3e6893f-a42f-40e7-8dd1-6b695a2474f1",
            "name": "2 Step Cleansing Kit",
            "description": "Short Description\nA convenient two-step cleansing kit designed to deeply cleanse and refresh the skin. It removes makeup, dirt, and impurities, leaving the skin clean, smooth, and hydrated.\n\nWhat It Does & Benefits\nThis kit effectively cleanses and nourishes the skin, preparing it for the next steps in your skincare routine. It provides a thorough cleanse while maintaining skin hydration and balance.\n\nInstructions\nStep 1: Apply the cleansing oil to dry skin and massage gently to dissolve makeup and impurities.\nStep 2: Follow with the foaming cleanser to remove any remaining residue. Rinse thoroughly with water.\n\nActive Ingredients\nCleansing Oil: Jojoba Oil, Grape Seed Oil\nFoaming Cleanser: Aloe Vera, Vitamin E, Salicylic Acid\n\nFull List of Ingredients (INCI)\nCleansing Oil: Jojoba Oil, Grape Seed Oil, Tocopheryl Acetate, Glycerin, Butylene Glycol, Phenoxyethanol, Ethylhexylglycerin\nFoaming Cleanser: Aloe Vera Extract, Vitamin E, Salicylic Acid, Glycerin, Sodium Chloride, Sodium Lauryl Sulfate, Phenoxyethanol, Ethylhexylglycerin\n\nSkin Type\nSuitable for all skin types, including sensitive skin. Ideal for normal to oily skin types.",
            "category_name": "vine vera",
            "subcategory_name": "Merlot",
            "price": 158,
            "quantity_value": 1,
            "quantity_unit": "unts"
        },
    ]

    services = [
        {
            "id": "36dd52d9-38cb-4625-8ce2-ef123303a84b",
            "name": "24K Cleopatra Regime facial Treatment",
            "description": "Short Description \nNone\n\nWhat It Does & Benefits\nRemoves impurities: The treatment cleanses and exfoliates the skin, removing dead skin cells, dirt, and excess oils.\nStimulates collagen production: 24K gold encourages collagen synthesis, improving skin elasticity and reducing fine lines and wrinkles.\nImproves skin texture: Provides smoother, softer, and more even skin tone.\nHydrates and nourishes: Deeply hydrates the skin, leaving it plump, moisturized, and rejuvenated.\nBrightens and evens complexion: The treatment works to enhance skin radiance and promote a more even skin tone.\nAnti-inflammatory effects: Gold particles in the treatment have anti-inflammatory properties, soothing irritated skin.\n\nInstructions\nCleanse: Begin with a deep cleanse to remove makeup, dirt, and impurities from the skin.\nExfoliate: Apply an exfoliating scrub or mask to slough off dead skin cells and prepare the skin for better absorption of the active ingredients.\nApply 24K Gold Serum: Gently massage a serum infused with 24K gold into the skin to stimulate collagen production and increase skin elasticity.\nFacial Massage: Use upward strokes to massage the product into the skin, promoting relaxation, circulation, and better product absorption.\nHydrating Mask: Apply a hydrating mask to nourish and lock in moisture. Leave on for 10-15 minutes.\nFinishing Serum and Moisturizer: Follow up with a finishing serum and moisturizer to seal in hydration and provide long-lasting results.\nSunscreen (if daytime): Apply sunscreen to protect the skin from UV damage and maintain the benefits of the treatment.\n\nActive Ingredients\n24K Gold: Enhances collagen production, reduces signs of aging, and promotes a glowing complexion.\nHyaluronic Acid: Provides deep hydration, plumping the skin and smoothing fine lines.\nVitamin C: Brightens the skin and reduces dark spots while offering antioxidant protection.\nPeptides: Help in restoring skin elasticity and reducing wrinkles.\nAloe Vera: Soothes and calms the skin, reducing redness and inflammation.\nRetinol: Stimulates skin renewal, reducing fine lines and improving skin texture.\n\nSkin Type\nSuitable for all skin types, especially mature, dry, and sensitive skin. Ideal for individuals seeking anti-aging benefits, hydration, and a radiant complexion.",
            "category_name": "Signature Facial Treatments",
            "subcategory_name": "Bespoke Facials",
            "price": 980,
            "duration": 75,
            "recovery_time": 15
        },
    ]

    try:
        print("Upserting products and services")
        await products_index_instance.upsert_documents(documents=products)
        await services_index_instance.upsert_documents(documents=services)

        products_document_count = await products_index_instance.get_document_count()
        services_document_count = await services_index_instance.get_document_count()

        print(f"Количество документов в продуктах: {products_document_count}")
        print(f"Количество документов в услугах: {services_document_count}")

    finally:
        await dependencies.close()


if __name__ == '__main__':
    asyncio.run(main())
