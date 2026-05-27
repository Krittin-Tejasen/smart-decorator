from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

app = FastAPI()


@app.get("/")
def root():

    return {
        "message": "Smart Decorator API Running"
    }


@app.post("/generate-room")
async def generate_room(

    room_type: str = Form(...),

    theme: str = Form(...),

    image: UploadFile = File(...)
):

    print(room_type)
    print(theme)
    print(image.filename)

    return {

        "generated_image":
            "fake_generated_room",

        "products": [

            {
                "id": "1",
                "name": "Modern Sofa",
                "imageUrl": "sofa",
                "price": 12990
            },

            {
                "id": "2",
                "name": "Minimal Lamp",
                "imageUrl": "lamp",
                "price": 2490
            }
        ]
    }