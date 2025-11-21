from django import forms

class UserRegistrationForm(forms.Form):
    username = forms.CharField(
        label="Имя пользователя",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    user_type = forms.ChoiceField(
        label="Тип пользователя",
        choices=[('ordinary', 'Обычный пользователь'), ('admin', 'Администратор')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    password = forms.CharField(
        label="Пароль",
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )
    password_confirm = forms.CharField(
        label="Подтвердите пароль",
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")

        if password and password_confirm and password != password_confirm:
            raise forms.ValidationError("Пароли не совпадают")
        return cleaned_data

# Оставьте TaskSubmissionForm ниже без изменений
class TaskSubmissionForm(forms.Form):
    filename = forms.CharField(
        label="Название файла (задачи)",
        max_length=255,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    python_file = forms.FileField(
        label="Файл .py",
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    python_code = forms.CharField(
        label="Или вставьте код Python сюда",
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 10})
    )


# webPart/accounts/forms.py

# ... (предыдущие импорты и классы)

class UserProfileForm(forms.Form):
    bio = forms.CharField(
        label="О себе",
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Расскажите немного о себе...'})
    )
    phone = forms.CharField(
        label="Телефон",
        required=False,
        max_length=20,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '+7 (999) 000-00-00'})
    )
    city = forms.CharField(
        label="Город",
        required=False,
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    birth_date = forms.DateField(
        label="Дата рождения",
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )