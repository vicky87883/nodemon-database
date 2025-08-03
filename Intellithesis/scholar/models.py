from django.db import models

class ResearchPaper(models.Model):
    TITLE_CHOICES = [
        ('RESEARCH_PAPER', 'Research Paper'),
        ('THESIS', 'Thesis'),
        ('JOURNAL', 'Journal'),
    ]

    file = models.FileField(upload_to='academic_content/')
    title = models.CharField(max_length=255, blank=True, null=True)
    authors = models.CharField(max_length=500, blank=True, null=True)
    abstract = models.TextField(blank=True, null=True)
    introduction = models.TextField(blank=True, null=True)
    methodology = models.TextField(blank=True, null=True)
    results = models.TextField(blank=True, null=True)
    conclusion = models.TextField(blank=True, null=True)
    references = models.TextField(blank=True, null=True)
    content_type = models.CharField(max_length=50, choices=TITLE_CHOICES, default='RESEARCH_PAPER')
    extracted_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or "Untitled Paper"