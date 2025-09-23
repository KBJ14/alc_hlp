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
        required=False,
        widget=forms.ClearableFileInput(attrs={"accept": "image/*"})
    )

    # After running Get Box, we pass the boxed image path back in here
    processed_image_path = forms.CharField(
        required=False,
        widget=forms.HiddenInput()
    )



