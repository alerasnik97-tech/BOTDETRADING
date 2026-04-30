import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailSender:
    def __init__(self):
        self.host = os.environ.get("MANIPULANTE_EMAIL_SMTP_HOST")
        self.port = os.environ.get("MANIPULANTE_EMAIL_SMTP_PORT")
        self.user = os.environ.get("MANIPULANTE_EMAIL_USER")
        self.password = os.environ.get("MANIPULANTE_EMAIL_PASSWORD")
        self.to = os.environ.get("MANIPULANTE_EMAIL_TO")
        self.enabled = all([self.host, self.port, self.user, self.password, self.to])

    def send_email(self, subject, body, dry_run=False):
        if not self.enabled:
            return {"status": "EMAIL_NOT_CONFIGURED"}

        if dry_run:
            print(f"[DRY-RUN] Email Subject: {subject}")
            return {"status": "DRY_RUN_OK"}

        msg = MIMEMultipart()
        msg['From'] = self.user
        msg['To'] = self.to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(self.host, int(self.port)) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            return {"status": "SENT"}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    sender = EmailSender()
    if not sender.enabled:
        print("EMAIL_NOT_CONFIGURED")
    else:
        print("EMAIL_READY")
