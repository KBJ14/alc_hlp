from django import forms

class VisionPromptForm(forms.Form):
    base_prompt = forms.CharField(
        label="Prompt",
        widget=forms.Textarea(attrs={"rows": 8})
    )
    query = forms.CharField(
        label="Text Prompt",
        widget=forms.Textarea(attrs={"rows": 4})
    )
    image = forms.ImageField(
        label="Image (PNG or JPG)",
        widget=forms.ClearableFileInput(attrs={"accept": "image/*"})
    )



