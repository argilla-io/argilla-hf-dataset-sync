FROM python:3.10

RUN apt-get update && apt-get -y install cron
RUN pip install argilla

COPY refresh.py refresh.py
COPY refresh-cron /etc/cron.d/refresh-cron

RUN chmod 0644 /etc/cron.d/refresh-cron
RUN crontab /etc/cron.d/refresh-cron

RUN touch /var/log/cron.log

CMD printenv > /etc/environment && cron && tail -f /var/log/cron.log
