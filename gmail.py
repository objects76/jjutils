import smtplib, os
import mimetypes
from datetime import datetime
from email.message import EmailMessage

def send_gmail(receiver_email:str,*,
               subject:str = "", body:str = "", files:list[str]=[]):
    sender_email = "objects76@gmail.com"
    app_password = os.environ.get("OBJECTS76_APP_PASSWORD",None)

    if not body:
        body = f"\nAt {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    if not subject:
        subject = f"Notiy from {sender_email} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    try:
      for file_path in files:
          mime_type, _ = mimetypes.guess_type(file_path)
          mime_type, mime_subtype = mime_type.split('/')
          # Open the file in binary mode
          with open(file_path, 'rb') as file:
              msg.add_attachment(file.read(),
                              maintype=mime_type,
                              subtype=mime_subtype,
                              filename=file.name)

      with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
          smtp.login(sender_email, app_password)
          smtp.send_message(msg)
      print("Email sent successfully!")
    except Exception as e:
      print(f"Error sending email: {e}")

# send_gmail('jjkim@rsupport.com', files=["n0..mp3", "n1..mp3"])