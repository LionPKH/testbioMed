from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User, AdminProfile, OrdinaryUserProfile


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    user_type = forms.ChoiceField(choices=User.USER_TYPE_CHOICES, initial='ordinary')

    class Meta:
        model = User
        fields = ('username', 'email', 'user_type', 'password1', 'password2')


class AdminProfileForm(forms.ModelForm):
    class Meta:
        model = AdminProfile
        fields = ('department', 'phone', 'permissions_level', 'access_code')


class OrdinaryUserProfileForm(forms.ModelForm):
    class Meta:
        model = OrdinaryUserProfile
        fields = ('bio', 'avatar', 'birth_date', 'phone', 'city', 'subscription_active')
        widgets = {
            'birth_date': forms.DateInput(attrs={'type': 'date'}),
            'bio': forms.Textarea(attrs={'rows': 4}),
        }


class TaskSubmissionForm(forms.Form):
    """
    Форма для отправки новой задачи с Python-кодом.
    """
    filename = forms.CharField(
        label="Название файла",
        required=True,
        max_length=255,
        initial="script.py",
        widget=forms.TextInput(attrs={'placeholder': 'script.py'}),
        help_text="Укажите название файла (с расширением .py)"
    )
    
    python_code = forms.CharField(
        label="Python код",
        required=False,
        widget=forms.Textarea(attrs={
            'placeholder': 'Напишите ваш Python код здесь...\n\nНапример:\ndef calculate():\n    return sum(range(100))\n\nresult = calculate()\nprint(result)',
            'rows': 15,
            'style': 'font-family: monospace; font-size: 14px;'
        }),
        help_text="Введите Python код или загрузите файл ниже."
    )
    
    python_file = forms.FileField(
        label="Или загрузите Python файл",
        required=False,
        widget=forms.FileInput(attrs={'accept': '.py'}),
        help_text="Загрузите готовый .py файл (необязательно, если код введен выше)"
    )
    
    def clean(self):
        cleaned_data = super().clean()
        python_code = cleaned_data.get('python_code')
        python_file = cleaned_data.get('python_file')
        filename = cleaned_data.get('filename')
        
        # Проверяем, что хотя бы один источник кода указан
        if not python_code and not python_file:
            raise forms.ValidationError("Необходимо либо ввести код, либо загрузить файл.")
        
        # Проверяем расширение файла
        if filename and not filename.endswith('.py'):
            raise forms.ValidationError("Название файла должно иметь расширением .py")
        
        # Если загружен файл, проверяем его расширение
        if python_file:
            if not python_file.name.endswith('.py'):
                raise forms.ValidationError("Можно загружать только .py файлы")
        
        return cleaned_data