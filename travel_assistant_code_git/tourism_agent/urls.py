# urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from .views import *

urlpatterns = [
    path('chat/', chat_with_graph_agent, name='chat_with_graph_agent'),
    path('safe-pdf/<str:filename>/', safe_pdf_view, name='safe_pdf'),
    path('upload-json/', batch_qa_evaluation, name='upload_json'),
]



if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)