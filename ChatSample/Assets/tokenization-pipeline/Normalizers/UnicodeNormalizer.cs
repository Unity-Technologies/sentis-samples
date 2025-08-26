using System.Text;

namespace Unity.ML.Tokenization.Normalizers
{
    public class UnicodeNormalizer : INormalizer
    {
        readonly NormalizationForm m_Form;

        public UnicodeNormalizer(NormalizationForm form = NormalizationForm.FormC) => m_Form = form;

        public SubString Normalize(SubString input) => input.ToString().Normalize(m_Form);
    }
}
