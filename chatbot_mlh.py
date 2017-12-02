import telebot
import requests as rq
import random
from telebot import types

token = "497481331:AAGeEz2U8p2yCe0Sx-YDE2kwKAm-2m0yZoY"
bot = telebot.TeleBot(token=token)
phrases = {}
stickers = []
users = {}


def parsing(urs):
    result = rq.get(urs)

    json_text = result.json()
    names = [[], []]
    k = 1
    if json_text['total'] != 0:
        for i in json_text['values']:
            names[0].append(str(k) + '. ' + i['name'])
            if "address" in i["location"].keys():
                names[1].append(i["location"]["address"])
            else:
                names[1].append(i["location"]["city"])
            k += 1

    return names
@bot.message_handler(commands=['start'])
def h(message):
    user = message.chat.id
    bot.send_message(user,"Привет, введи город и я расскажу и покажу события вокруг тебя")

@bot.message_handler(content_types=['text'])

def remind(message):
    user = message.chat.id
    phrases[user] = message.text

    bot.send_message(user, "Введите город")

    urs = 'https://api.timepad.ru/v1/events.json?limit=20&skip=0&cities='



    for user in phrases.keys():
        urs += phrases[user]
        urs += '&fields=location&sort=+starts_at'
        tmp = parsing(urs)
        for i, j in zip(tmp[0], tmp[1]):
            bot.send_message(user, i)
            button(message,j)
        map_pic = pic(urs + '&fields=location&sort=+starts_at')
        print(urs)
        if map_pic != 'Мы ничего не нашли':
            bot.send_photo(user, map_pic)

        else:
            bot.send_message(user, map_pic)

def button(message, text):

    keyboard = types.InlineKeyboardMarkup()
    url_button = types.InlineKeyboardButton(text="Перейти на Яндекс", url="https://yandex.ru/maps/213/moscow/?mode=search&ll=37.528199%2C55.789735&z=11&text=["+ text +"]&sll=37.557772%2C55.825831&sspn=0.157242%2C0.052373")
    keyboard.add(url_button)
    bot.send_message(message.chat.id, "Ссылка на карты", reply_markup=keyboard)


def cord_l(a, i):
    return str(a[1]) + ',' + str(a[0]) + ",pmwtm" + str(i + 1)


def pic(site):
    htt = site
    val = rq.get(htt)
    rec = 'Мы ничего не нашли'
    json_text = val.json()
    if json_text['total'] != 0:
        tx = json_text['values']

        points = "&size=450,450&z=10&l=map&pt="

        center = [[float(tx[0]['location']['coordinates'][0]),
                   float(tx[0]['location']['coordinates'][1])]]
        points += cord_l(center[-1], 0)
        for i in range(1, len(tx)):
            if "coordinates" in tx[i]['location'].keys():
                center.append([float(tx[i]['location']['coordinates'][0]),
                               float(tx[i]['location']['coordinates'][1])])
                points += '~' + cord_l(center[-1], i)

        w1, w2 = 0, 0
        for i in center:
            w1 += i[0]
            w2 += i[1]
        w1 = w1 / len(center)
        w2 = w2 / len(center)
        rec = "https://static-maps.yandex.ru/1.x/?ll=" + str(w2)[:9] + ',' + str(w1)[:9] + points
    return rec


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    if call.message:
        id = call.message.chat.id
        if call.data == "da":
            send_taxi(call.message)
            # if id not in users.keys():
            #    send_error(call.message)
            # bot.send_message(id,random.choice())


def send_error(message):
    bot.send_message(message.chat.id, "Мы чет попутали((")


@bot.message_handler(content_types=['sticker'])
def echo(message):
    user = message.chat.id
    sticker = message.sticker.file_id

    stickers.append(sticker)

    bot.send_sticker(user, random.choice(stickers))


bot.polling(none_stop=True)