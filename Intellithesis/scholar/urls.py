# scholar/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_academic_content, name='upload_academic_content'),
    path('list/', views.AcademicContentList.as_view(), name='academic_content_list'),
    path('caption-match/', views.process_caption_matching_request, name='caption_matching'),
    path('image-description/', views.process_image_description_request, name='image_description'),
    path('caption-normalize/', views.process_caption_normalization_request, name='caption_normalization'),
    path('caption-validate/', views.process_caption_validation_request, name='caption_validation'),
    path('ocr-process/', views.process_ocr_document_request, name='ocr_processing'),
]
