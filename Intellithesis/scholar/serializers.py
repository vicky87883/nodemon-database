from rest_framework import serializers
from .models import ResearchPaper

class ResearchPaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchPaper
        fields = ['id', 'file', 'title', 'authors', 'abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references', 'content_type', 'extracted_date']