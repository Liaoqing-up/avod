from avod.core.feature_extractors.bev_vgg import BevVgg
from avod.core.feature_extractors.bev_vgg_pyramid import BevVggPyr
from avod.core.feature_extractors.bev_resnet_pyramid import BevResPyr

from avod.core.feature_extractors.img_vgg import ImgVgg
from avod.core.feature_extractors.img_vgg_pyramid import ImgVggPyr
from avod.core.feature_extractors.img_resnet_pyramid import ImgResPyr



def get_extractor(extractor_config):

    extractor_type = extractor_config.WhichOneof('feature_extractor')

    # BEV feature extractors
    if extractor_type == 'bev_vgg':
        return BevVgg(extractor_config.bev_vgg)
    elif extractor_type == 'bev_vgg_pyr':
        return BevVggPyr(extractor_config.bev_vgg_pyr)
    elif extractor_type == 'bev_res_pyr':
        return BevResPyr(extractor_config.bev_res_pyr)

    # Image feature extractors
    elif extractor_type == 'img_vgg':
        return ImgVgg(extractor_config.img_vgg)
    elif extractor_type == 'img_vgg_pyr':
        return ImgVggPyr(extractor_config.img_vgg_pyr)
    elif extractor_type == 'img_res_pyr':
        return ImgResPyr(extractor_config.img_res_pyr)

    return ValueError('Invalid feature extractor type', extractor_type)
