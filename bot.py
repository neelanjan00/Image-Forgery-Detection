import keras
from keras import backend as K
from telegram.chataction import ChatAction
from telegram.files.document import Document
import modelCore
import tensorflow as tf
import matplotlib.pyplot as plt
import base64
import os

import numpy as np
from cv2 import cv2
import logging
from typing import List

from telegram import Update,Message
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.files.file import File
from telegram.files.photosize import PhotoSize

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def decode_an_image_array(rgb, dn=1):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)[:, ::dn, ::dn]
    K.clear_session()
    manTraNet = modelCore.load_trained_model()
    return manTraNet.predict(x)[0, ..., 0]


def decode_an_image_file(image_file, dn=1):
    mask = decode_an_image_array(image_file, dn)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_file[::dn, ::dn])
    plt.imshow(mask, cmap='jet', alpha=.5)
    plt.savefig('h.png', bbox_inches='tight', pad_inches=-0.1)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def predict(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    if update.message.text:
        update.message.reply_text(update.message.text)
    message:Message = update.message
    photo:List[PhotoSize] = message.photo
    document:Document = message.document
    context.bot.send_chat_action(chat_id=message.chat.id,action=ChatAction.UPLOAD_PHOTO,timeout=60000)
    if photo:
        p:PhotoSize = photo[-1]
        file:File = p.get_file(timeout=10000)
        arr:bytearray = file.download_as_bytearray()
        nparr = np.frombuffer(arr,np.uint8)
        inp_img = cv2.imdecode(np.frombuffer(nparr, np.uint8), cv2.IMREAD_UNCHANGED)        
        decode_an_image_file(inp_img)
        output = cv2.imread('h.png')
        _, outputBuffer = cv2.imencode('.jpg', output)
        OutputBase64String = base64.b64encode(outputBuffer).decode('utf-8')
        message.reply_photo(photo=open('h.png','rb'))
    elif document:
        file:File = document.get_file(timeout=10000)
        arr:bytearray = file.download_as_bytearray()
        nparr = np.frombuffer(arr,np.uint8)
        inp_img = cv2.imdecode(np.frombuffer(nparr, np.uint8), cv2.IMREAD_UNCHANGED)        
        decode_an_image_file(inp_img)
        output = cv2.imread('h.png')
        _, outputBuffer = cv2.imencode('.jpg', output)
        OutputBase64String = base64.b64encode(outputBuffer).decode('utf-8')
        message.reply_photo(photo=open('h.png','rb'))
    else:
        pass
        


def main():
    """Start the bot."""

    updater = Updater("1307258507:AAEEcGsr2hnCEy5kQxckVxmVuR7pgZ9ZbC0", use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(~Filters.command, predict))

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()