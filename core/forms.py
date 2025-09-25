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

    # Raw uploaded image (saved path in media) to allow further processing steps
    raw_image_path = forms.CharField(
        required=False,
        widget=forms.HiddenInput()
    )

    # Segmented (mask overlay) image path
    segmented_image_path = forms.CharField(
        required=False,
        widget=forms.HiddenInput()
    )



